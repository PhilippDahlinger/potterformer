import torch



class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.qkv_matrix = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_matrix = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "num_heads must divide hidden_dim equally"
        self.num_heads = num_heads
        self.head_latent_dim = hidden_dim // num_heads
        self.ln1 = torch.nn.LayerNorm(self.hidden_dim)
        self.ln2 = torch.nn.LayerNorm(self.hidden_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4*hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*hidden_dim, hidden_dim)
        )


    def split_to_heads(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_latent_dim)
        x = torch.transpose(x, dim0=1, dim1=2)
        # output has shape (batch_size, num_heads, context_size, head_latent_dim)
        return x

    def merge_from_heads(self, x):
        x = torch.transpose(x, dim0=1, dim1=2)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        return x

    def build_mask(self, context_size, device):
        t = torch.ones(context_size, context_size)
        t = torch.torch.triu(t, diagonal=1)
        # replace 1s with very low value
        t = t.masked_fill(t == 1, torch.finfo(t.dtype).min)
        return t.to(device)

    def attention(self, x):
        (batch_size, context_size, d) = x.shape
        assert d == self.hidden_dim, "Hidden dim mismatch"
        qkv = self.qkv_matrix(x)
        queries = self.split_to_heads(qkv[:, :, 0:d])
        keys = self.split_to_heads(qkv[:, :, d:2*d])
        values = self.split_to_heads(qkv[:, :, 2*d:])

        # Q * K^T
        queries_key = torch.einsum('bnij,bnkj->bnik', queries, keys)
        # normalization
        queries_key = queries_key / (self.head_latent_dim ** 0.5)
        # add masking
        queries_key_masked = queries_key + self.build_mask(queries_key.shape[3], device=queries_key.device)
        # softmax
        softmax_output = torch.softmax(queries_key_masked, dim=-1)
        # weighted sum
        output = torch.einsum("bnik,bnkd->bnid", softmax_output, values)
        # merge
        output = self.merge_from_heads(output)
        # exchange infos between heads
        output = self.output_matrix(output)
        return output

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x




class Model(torch.nn.Module):
    def __init__(self, vocab_size, context_size, hidden_dim=256, num_heads=8):
        super(Model, self).__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = torch.nn.Embedding(context_size, hidden_dim)
        self.transformer = torch.nn.Sequential(
            TransformerBlock(hidden_dim, num_heads),
            TransformerBlock(hidden_dim, num_heads),
            TransformerBlock(hidden_dim, num_heads),
            TransformerBlock(hidden_dim, num_heads),
            TransformerBlock(hidden_dim, num_heads),
            TransformerBlock(hidden_dim, num_heads)
        )
        self.final_ln = torch.nn.LayerNorm(hidden_dim)




    def forward(self, x):
        token_embedding = self.token_embedding(x)
        pos_embedding = self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        embedding = token_embedding + pos_embedding
        processed = self.final_ln(self.transformer(embedding))
        # get vocab output prob
        vocab_dist = torch.nn.functional.linear(processed, self.token_embedding.weight)
        return vocab_dist