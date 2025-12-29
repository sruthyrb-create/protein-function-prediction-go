import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinCNN(nn.Module):
    def __init__(
        self,
        vocab_size=21,      # 20 amino acids + padding
        embed_dim=128,
        num_filters=256,
        kernel_sizes=(3, 5, 7),
        num_labels=9494
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(
            num_filters * len(kernel_sizes),
            num_labels
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        x = self.embedding(x)          # (B, L, D)
        x = x.transpose(1, 2)          # (B, D, L)

        conv_outs = []
        for conv in self.convs:
            z = F.relu(conv(x))        # (B, F, L')
            z = F.max_pool1d(z, z.size(2)).squeeze(2)
            conv_outs.append(z)

        h = torch.cat(conv_outs, dim=1)
        logits = self.fc(h)

        return torch.sigmoid(logits)
