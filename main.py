import torch

from language_dataset import LanguageDataset
from lit_module import LitModel
import pytorch_lightning as pl
from tokenization import char_tokenize


def main():
    input_file = "data/processed/hp1.txt"
    # load the cleaned data
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()
    # tokenize the text
    token_ids, vocab, char_to_id, id_to_char = char_tokenize(text)
    ds = LanguageDataset(token_ids, context_size=128)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    lit_model = LitModel(vocab_size=len(vocab), context_size=128, hidden_dim=256)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",  # later gpu
        devices=1,
        precision="32-true",  # FP32
        gradient_clip_val=1.0,  # Gradient Clipping
        log_every_n_steps=10,
    )

    # Train the model, no validation set for now
    trainer.fit(lit_model, dl, dl)


if __name__ == "__main__":
    main()