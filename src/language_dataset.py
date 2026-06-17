import torch.utils.data


class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, context_size=128):
        self.token_ids = token_ids
        self.context_size = context_size

    def __len__(self):
        return len(self.token_ids) - self.context_size - 1

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx + self.context_size]
        y = self.token_ids[idx + 1:idx + self.context_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

