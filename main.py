import torch

from language_dataset import LanguageDataset
from model import Model
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

    model = Model(len(vocab), context_size=128, hidden_dim=256)

    for x, y in dl:
        out = model(x)
        break


if __name__ == "__main__":
    main()