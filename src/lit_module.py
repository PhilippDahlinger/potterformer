import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model import Model  # deine bestehende Model-Klasse


class LitModel(pl.LightningModule):
    def __init__(self, vocab_size, context_size, hidden_dim=256, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(vocab_size, context_size, hidden_dim, num_heads=8)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        if stage == "val":
            print("Validation step not implemented yet.")
            return -1.0
        x, y = batch
        logits = self.model(x)  # erwartet (B, T, vocab_size)
        # CrossEntropy erwartet (N, C) und (N,), daher flach machen
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # no val implemented
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)