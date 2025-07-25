import torch
import torch.nn as nn
import tiktoken

class TokenEmbedder:
    def __init__(self, tokenizer_name: str, vocab_size: int, d_model: int, context_length: int):
        self.context_length = context_length
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        self.token_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.pos_embedding_layer = nn.Embedding(context_length, d_model)

    def process_text(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)

        if len(token_ids) > self.context_length:
            token_ids = token_ids[:self.context_length]
        else:
            pad_token = self.tokenizer.eot_token if hasattr(self.tokenizer, 'eot_token') else 0
            token_ids = token_ids + [pad_token] * (self.context_length - len(token_ids))
        
        input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        token_embeds = self.token_embedding_layer(input_tensor) 
        pos_embeds = self.pos_embedding_layer(torch.arange(self.context_length))
        
        final_embeddings = token_embeds + pos_embeds
        
        return final_embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.K = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.V = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_model)

        context_vecs = self.out_proj(context_vec)
        
        return context_vecs