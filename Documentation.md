Date:10/20/2025

Project Name: "Protein function prediction"



Project flow:

src:

---->go\_utils.py

---->data\_loader.py





data:

---->sample\_submission.tsv

---->IA.tsv



---->train

-------->train\_terms.tsv

-------->train\_sequences.fasta

-------->train\_taxonomy.tsv

-------->go-basic.obo



---->test

-------->testsuperset.fasta

-------->testsuperset-taxon-list.tsv



Commands executed:



Anaconda Prompt

Installations:\[Inside the project file]



\#Environment creation

conda create -n pfpc python=3.10 -y

conda activate pfpc



\#Essentials Installations

conda install -y pip

pip install numpy pandas scikit-learn tqdm obonet biopython joblib



Descriptions:

A Conda environment keeps dependencies isolated so notebook runs and scripts are reproducible. The packages:

**--obonet** to parse OBO (GO) files into a graph (easy to use),

**--biopython** to parse FASTA robustly,

**--scikit-learn** for the MultiLabelBinarizer and some baseline models,

**--pandas/numpy** for table/array manipulation,

**--joblib** for saving/loading artifacts,

**--tqdm** shows progress bars (convenience).



\#Project scaffold

mkdir -p src outputs artifacts notebooks



Code:



src/go-utils.py



\# src/go\_utils.py

"""

GO utilities: parse the go-basic.obo file (OBO format) using obonet,

build a DAG with parent->child relations, and compute ancestors for terms.



Functions:

\- load\_go\_obo(path) -> networkx.MultiDiGraph (as returned by obonet.read\_obo)

\- build\_parent\_map(G) -> dict(term -> set(parents))

\- get\_ancestors(G, term) -> set(ancestors)

\- propagate\_terms(terms, ancestors\_map) -> set(terms + ancestors)

"""



import obonet

from collections import defaultdict



def load\_go\_obo(obo\_path):

&nbsp;   """

&nbsp;   Parse OBO file and return a networkx graph (obonet representation).

&nbsp;   Each node is a GO id like 'GO:0008150' and node attributes contain 'name' and 'namespace' etc.

&nbsp;   """

&nbsp;   G = obonet.read\_obo(obo\_path)

&nbsp;   return G



def build\_parent\_map(G, relation\_types=('is\_a', 'part\_of')):

&nbsp;   """

&nbsp;   Build a mapping: term -> set(parent\_terms)

&nbsp;   We consider 'is\_a' and 'part\_of' edges by default (common GO relations).

&nbsp;   obonet stores edges as (child, parent) for 'is\_a' with edge attribute 'relation' or 'is\_a' encoded in edges.

&nbsp;   """

&nbsp;   parent\_map = defaultdict(set)

&nbsp;   for u, v, data in G.edges(data=True):

&nbsp;       # In OBO/obonet, edges go from child -> parent.

&nbsp;       rel = data.get('relation') or data.get('type') or ''

&nbsp;       # Accept is\_a and part\_of by default. If relation is empty, treat as generic parent.

&nbsp;       if rel == '' or rel in relation\_types:

&nbsp;           parent\_map\[u].add(v)

&nbsp;   return parent\_map



def compute\_ancestors(parent\_map):

&nbsp;   """

&nbsp;   Given parent\_map: term -> set(parents), compute full ancestors for each term (transitive closure).

&nbsp;   Returns dict term -> set(all ancestors).

&nbsp;   Uses DFS / iterative approach; caches results for efficiency.

&nbsp;   """

&nbsp;   ancestors = {}

&nbsp;   def dfs(term, visited):

&nbsp;       if term in ancestors:

&nbsp;           return ancestors\[term]

&nbsp;       res = set()

&nbsp;       for p in parent\_map.get(term, ()):

&nbsp;           if p in visited:

&nbsp;               continue

&nbsp;           visited.add(p)

&nbsp;           res.add(p)

&nbsp;           res.update(dfs(p, visited))

&nbsp;           visited.remove(p)

&nbsp;       ancestors\[term] = res

&nbsp;       return res



&nbsp;   for t in list(parent\_map.keys()):

&nbsp;       dfs(t, set())



&nbsp;   # Ensure terms with no parents appear in map (empty set)

&nbsp;   # Also include nodes that have no outgoing edges (leaf nodes) if missing

&nbsp;   for t in list(parent\_map.keys()):

&nbsp;       ancestors.setdefault(t, set())



&nbsp;   return ancestors



def propagate\_terms(terms, ancestors\_map):

&nbsp;   """

&nbsp;   Given a set/list of terms (strings like 'GO:XXXXX'), return a new set that

&nbsp;   includes the original terms AND all their ancestor terms.

&nbsp;   """

&nbsp;   out = set()

&nbsp;   for t in terms:

&nbsp;       out.add(t)

&nbsp;       if t in ancestors\_map:

&nbsp;           out.update(ancestors\_map\[t])

&nbsp;   return out



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   # quick smoke test (run: python src/go\_utils.py)

&nbsp;   import sys, os

&nbsp;   if len(sys.argv) < 2:

&nbsp;       print("Usage: python src/go\_utils.py path/to/go-basic.obo")

&nbsp;       sys.exit(1)

&nbsp;   obo = sys.argv\[1]

&nbsp;   G = load\_go\_obo(obo)

&nbsp;   pm = build\_parent\_map(G)

&nbsp;   anc = compute\_ancestors(pm)

&nbsp;   print("Loaded GO graph: nodes=%d, edges=%d" % (len(G.nodes()), len(G.edges())))

&nbsp;   sample = list(anc.keys())\[:5]

&nbsp;   for s in sample:

&nbsp;       print(s, "->", len(anc\[s]), "ancestors")





Description:



* Why parse GO? GO (Gene Ontology) is a directed acyclic graph (DAG) where child terms imply parent terms biologically. For correct training \& evaluation we must respect this hierarchy.



* obonet.read\_obo returns nodes and edges. Edges are child → parent (that’s convenient for building ancestors).



* We compute ancestors\_map (transitive closure): for a term t, ancestors\_map\[t] includes all parent, grandparent, etc. This is used to propagate labels upward: if a protein is annotated with a specific function (child), it also implicitly has all more-general parent functions.



* We consider relations is\_a and part\_of — these are the standard hierarchical relations used to propagate annotations.





data\_loader.py



\# src/data\_loader.py

"""

Load sequences (FASTA) and annotation terms (train\_terms.tsv),

propagate annotations using GO ancestors, and build a MultiLabelBinarizer.



Produces:

\- proteins: list of protein IDs

\- sequences: list of sequences aligned with proteins

\- labels: list of lists of GO terms (propagated)

\- mlb object (fitted MultiLabelBinarizer)

\- saves artifacts: artifacts/mlb\_classes.npy and artifacts/protein\_index\_map.joblib

"""



import os

from collections import defaultdict

from Bio import SeqIO

import pandas as pd

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

import joblib



from src.go\_utils import load\_go\_obo, build\_parent\_map, compute\_ancestors, propagate\_terms



def read\_fasta(fasta\_path):

&nbsp;   """

&nbsp;   Return dict protein\_id -> amino acid sequence (string).

&nbsp;   Expects FASTA headers with IDs like sp|P9WHI7|RECN\_MYCT or simply P9WHI7; we'll extract the accession token.

&nbsp;   """

&nbsp;   prot2seq = {}

&nbsp;   for rec in SeqIO.parse(fasta\_path, "fasta"):

&nbsp;       header = rec.id  # biopython's SeqRecord.id is first token after '>'

&nbsp;       # header might be like 'sp|P9WHI7|RECN\_MYCT' -> extract accession as second field if pipe-separated

&nbsp;       if '|' in header:

&nbsp;           parts = header.split('|')

&nbsp;           if len(parts) >= 2:

&nbsp;               acc = parts\[1]

&nbsp;           else:

&nbsp;               acc = header

&nbsp;       else:

&nbsp;           acc = header

&nbsp;       seq = str(rec.seq).upper()

&nbsp;       prot2seq\[acc] = seq

&nbsp;   return prot2seq



def read\_train\_terms(train\_terms\_tsv):

&nbsp;   """

&nbsp;   Read train\_terms.tsv expecting columns: protein\_id \\t GO:xxxxx \\t Ontology (MFO/BPO/CCO or similar)

&nbsp;   Returns dict protein\_id -> set(terms)

&nbsp;   """

&nbsp;   df = pd.read\_csv(train\_terms\_tsv, sep='\\t', header=None, names=\['protein','go','ont'], dtype=str)

&nbsp;   ann = defaultdict(set)

&nbsp;   for \_, row in df.iterrows():

&nbsp;       pid = row\['protein']

&nbsp;       go = row\['go']

&nbsp;       if pd.isna(go) or not isinstance(go, str): 

&nbsp;           continue

&nbsp;       ann\[pid].add(go.strip())

&nbsp;   return ann



def build\_dataset(fasta\_path, train\_terms\_tsv, go\_obo\_path, save\_dir="artifacts"):

&nbsp;   # load GO and ancestors

&nbsp;   G = load\_go\_obo(go\_obo\_path)

&nbsp;   parent\_map = build\_parent\_map(G)

&nbsp;   ancestors\_map = compute\_ancestors(parent\_map)



&nbsp;   prot2seq = read\_fasta(fasta\_path)

&nbsp;   ann = read\_train\_terms(train\_terms\_tsv)



&nbsp;   proteins = \[]

&nbsp;   sequences = \[]

&nbsp;   labels\_list = \[]

&nbsp;   missing\_in\_fasta = 0

&nbsp;   for pid, gos in ann.items():

&nbsp;       if pid not in prot2seq:

&nbsp;           missing\_in\_fasta += 1

&nbsp;           continue

&nbsp;       propagated = propagate\_terms(gos, ancestors\_map)

&nbsp;       proteins.append(pid)

&nbsp;       sequences.append(prot2seq\[pid])

&nbsp;       labels\_list.append(sorted(list(propagated)))



&nbsp;   print("Total annotated proteins read:", len(ann))

&nbsp;   print("Proteins with FASTA available and used:", len(proteins))

&nbsp;   if missing\_in\_fasta:

&nbsp;       print("Proteins present in annotations but missing in FASTA:", missing\_in\_fasta)



&nbsp;   # build MultiLabelBinarizer (term -> index)

&nbsp;   mlb = MultiLabelBinarizer(sparse\_output=False)

&nbsp;   Y = mlb.fit\_transform(labels\_list)  # shape (n\_proteins, n\_terms)

&nbsp;   print("Number of unique GO terms (after propagation) in training:", len(mlb.classes\_))



&nbsp;   # save artifacts

&nbsp;   os.makedirs(save\_dir, exist\_ok=True)

&nbsp;   joblib.dump(mlb.classes\_, os.path.join(save\_dir, "mlb\_classes.npy"))

&nbsp;   joblib.dump({'proteins': proteins, 'index': {p:i for i,p in enumerate(proteins)}}, os.path.join(save\_dir, "protein\_index\_map.joblib"))

&nbsp;   np.save(os.path.join(save\_dir, "Y.npy"), Y, allow\_pickle=False)

&nbsp;   # save sequences mapping for later (maybe large); for now save a small csv

&nbsp;   pd.DataFrame({'protein':proteins, 'sequence':sequences}).to\_csv(os.path.join(save\_dir, "proteins\_sequences.csv"), index=False)



&nbsp;   return proteins, sequences, labels\_list, mlb, Y



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   import argparse

&nbsp;   parser = argparse.ArgumentParser()

&nbsp;   parser.add\_argument("--fasta", default="data/train\_sequences.fasta")

&nbsp;   parser.add\_argument("--terms", default="data/train\_terms.tsv")

&nbsp;   parser.add\_argument("--obo", default="data/go-basic.obo")

&nbsp;   parser.add\_argument("--out", default="artifacts")

&nbsp;   args = parser.parse\_args()



&nbsp;   proteins, sequences, labels\_list, mlb, Y = build\_dataset(args.fasta, args.terms, args.obo, save\_dir=args.out)

&nbsp;   print("Saved artifacts to", args.out)





(pfpc) C:\\Users\\rbsru\\Downloads\\Protein function prediction> python -m src.data\_loader --fasta data/train\_sequences.fasta --terms data/train\_terms.tsv --obo data/go-basic.obo --out artifacts

Total annotated proteins read: 82405

Proteins with FASTA available and used: 82404

Proteins present in annotations but missing in FASTA: 1

Number of unique GO terms (after propagation) in training: 29616

Saved artifacts to artifacts





Code explanation:



go-utils.py

---> **obonet module**: load obo ontology file into network x graph.Go distributed in obo format and obonet.read-obo give ready graph \[node: GO ID , edge relationship](child--->parent)



--->**defaultdict**: python standard library; normal dict with default value factory. default dict(set) ---> missing keys create empty set. 



--->**obonet.read\_ob**o()inbuild function in obonet to read file



--->iteration done on graph edges(child, parent)

---> in obonet edges go from child --->parent

---> edges include(is\_a or part\_of) filter to include typically meaningful hierarchy edges. Empty Relation = still accepted. (some obo edges omit relation still represent parent-child)



**Need to recursively follow parent-- map faster and simpler to use than repeated graph querying.** 




