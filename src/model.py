import torch


class Model(torch.nn.Module):
    def __init__(self, vocab_size, context_size, hidden_dim=256):
        super(Model, self).__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = torch.nn.Embedding(context_size, hidden_dim)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        pos_embedding = self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        embedding = token_embedding + pos_embedding
        print("stop")