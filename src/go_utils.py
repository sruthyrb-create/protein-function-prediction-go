 # src/go_utils.py
"""
GO utilities: parse the go-basic.obo file (OBO format) using obonet,
build a DAG with parent->child relations, and compute ancestors for terms.

Functions:
- load_go_obo(path) -> networkx.MultiDiGraph (as returned by obonet.read_obo)
- build_parent_map(G) -> dict(term -> set(parents))
- get_ancestors(G, term) -> set(ancestors)
- propagate_terms(terms, ancestors_map) -> set(terms + ancestors)
"""

import obonet
from collections import defaultdict

def load_go_obo(obo_path):
    """
    Parse OBO file and return a networkx graph (obonet representation).
    Each node is a GO id like 'GO:0008150' and node attributes contain 'name' and 'namespace' etc.
    """
    G = obonet.read_obo(obo_path)
    return G

def build_parent_map(G, relation_types=('is_a', 'part_of')):
    """
    Build a mapping: term -> set(parent_terms)
    We consider 'is_a' and 'part_of' edges by default (common GO relations).
    obonet stores edges as (child, parent) for 'is_a' with edge attribute 'relation' or 'is_a' encoded in edges.
    """
    parent_map = defaultdict(set)
    for u, v, data in G.edges(data=True):
        # In OBO/obonet, edges go from child -> parent.
        rel = data.get('relation') or data.get('type') or ''
        # Accept is_a and part_of by default. If relation is empty, treat as generic parent.
        if rel == '' or rel in relation_types:
            parent_map[u].add(v)
    return parent_map

def compute_ancestors(parent_map):
    """
    Given parent_map: term -> set(parents), compute full ancestors for each term (transitive closure).
    Returns dict term -> set(all ancestors).
    Uses DFS / iterative approach; caches results for efficiency.
    """
    ancestors = {}
    def dfs(term, visited):
        if term in ancestors:
            return ancestors[term]
        res = set()
        for p in parent_map.get(term, ()):
            if p in visited:
                continue
            visited.add(p)
            res.add(p)
            res.update(dfs(p, visited))
            visited.remove(p)
        ancestors[term] = res
        return res

    for t in list(parent_map.keys()):
        dfs(t, set())

    # Ensure terms with no parents appear in map (empty set)
    # Also include nodes that have no outgoing edges (leaf nodes) if missing
    for t in list(parent_map.keys()):
        ancestors.setdefault(t, set())

    return ancestors

def propagate_terms(terms, ancestors_map):
    """
    Given a set/list of terms (strings like 'GO:XXXXX'), return a new set that
    includes the original terms AND all their ancestor terms.
    """
    out = set()
    for t in terms:
        out.add(t)
        if t in ancestors_map:
            out.update(ancestors_map[t])
    return out

if __name__ == "__main__":
    # quick smoke test (run: python src/go_utils.py)
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python src/go_utils.py path/to/go-basic.obo")
        sys.exit(1)
    obo = sys.argv[1]
    G = load_go_obo(obo)
    pm = build_parent_map(G)
    anc = compute_ancestors(pm)
    print("Loaded GO graph: nodes=%d, edges=%d" % (len(G.nodes()), len(G.edges())))
    sample = list(anc.keys())[:5]
    for s in sample:
        print(s, "->", len(anc[s]), "ancestors")
