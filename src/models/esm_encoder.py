import torch
import esm

class ESMEncoder(torch.nn.Module):
    def __init__(self, model_name="esm2_t6_8M_UR50D"):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, sequences):
        data = [(str(i), seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)

        with torch.no_grad():
            out = self.model(tokens, repr_layers=[6])
            reps = out["representations"][6]

        # Mean pooling (ignore padding)
        mask = tokens != self.alphabet.padding_idx
        pooled = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        return pooled
