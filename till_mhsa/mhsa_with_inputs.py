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
        print(f"Input text: '{text}'")
        
        token_ids = self.tokenizer.encode(text)
        print(f"Raw token IDs: {token_ids}")
        print(f"Number of tokens: {len(token_ids)}")

        if len(token_ids) > self.context_length:
            token_ids = token_ids[:self.context_length]
            print(f"Truncated token IDs: {token_ids}")
        else:
            pad_token = self.tokenizer.eot_token if hasattr(self.tokenizer, 'eot_token') else 0
            original_length = len(token_ids)
            token_ids = token_ids + [pad_token] * (self.context_length - len(token_ids))
            print(f"Padded token IDs: {token_ids}")
            print(f"Padding: {self.context_length - original_length} tokens with value {pad_token}")
        
        input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        print(f"Input tensor shape: {input_tensor.shape}")

        token_embeds = self.token_embedding_layer(input_tensor)
        print(f"Token embeddings shape: {token_embeds.shape}")
        
        pos_embeds = self.pos_embedding_layer(torch.arange(self.context_length))
        print(f"Position embeddings shape: {pos_embeds.shape}")
        
        final_embeddings = token_embeds + pos_embeds
        print(f"Final embeddings shape: {final_embeddings.shape}")
        
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
        print(f"\n------------------ MULTI-HEAD SELF-ATTENTION ------------------")
        b, num_tokens, d_in = x.shape
        print(f"Input shape: {x.shape}")
        
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)
        print(f"Q, K, V shapes: {queries.shape}")

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        print(f"Reshaped Q, K, V: {queries.shape}")

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        print(f"Transposed Q, K, V: {queries.shape}")

        attn_scores = queries @ keys.transpose(2, 3)
        print(f"Attention scores shape: {attn_scores.shape}")
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        print(f"Mask shape: {mask_bool.shape}")
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        print(f"Applied causal mask")

        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        print(f"Attention weights shape: {attn_weights.shape}")
        
        attn_weights = self.dropout(attn_weights)
        print(f"Applied dropout")

        context_vec = attn_weights @ values
        print(f"Context vector shape: {context_vec.shape}")
        
        context_vec = context_vec.transpose(1, 2)
        print(f"Transposed context vector: {context_vec.shape}")
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_model)
        print(f"Flattened context vector: {context_vec.shape}")

        context_vecs = self.out_proj(context_vec)
        print(f"Output projection shape: {context_vecs.shape}")

        return context_vecs 

def get_input_with_default(prompt: str, default_value, convert_func=None):
    user_input = input(prompt).strip()
    if not user_input:
        return default_value
    
    if convert_func:
        try:
            return convert_func(user_input)
        except ValueError:
            print(f"Invalid input, using default: {default_value}")
            return default_value
    return user_input

def validate_parameters(tokenizer_name, context_length, dropout, d_model, num_heads):
    try:
        tiktoken.get_encoding(tokenizer_name)
    except Exception:
        print(f"Invalid tokenizer '{tokenizer_name}', using default 'gpt2'")
        tokenizer_name = "gpt2"
    
    if context_length <= 0:
        print("Context length must be positive, using default 64")
        context_length = 64
    
    if not 0 <= dropout <= 1:
        print("Dropout must be between 0 and 1, using default 0.1")
        dropout = 0.1
    
    if d_model % num_heads != 0:
        print(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        return None, None, None
    
    return tokenizer_name, context_length, dropout

def main():
    TOKENIZER_NAME = "gpt2"
    VOCAB_SIZE = 50257   
    CONTEXT_LENGTH = 64   
    D_MODEL = 256          
    NUM_HEADS = 8          
    DROPOUT = 0.1
    
    print("Welcome to MHSA From Scratch!")
    
    input_text = input("\nInput a text for which you want to find context_vector: ").strip()
    if not input_text:
        print("No input text provided. Exiting.")
        return
    
    tokenizer_name = get_input_with_default(
        f"Do you want to use any specific tokenizer? (Default: {TOKENIZER_NAME}) ",
        TOKENIZER_NAME
    )
    
    context_length = get_input_with_default(
        f"Do you have any specific context length in mind? (Default: {CONTEXT_LENGTH}) ",
        CONTEXT_LENGTH,
        int
    )
    
    dropout = get_input_with_default(
        f"Do you have any specific dropout value in mind? (Default: {DROPOUT}) ",
        DROPOUT,
        float
    )
    
    print(f"\n------------------ CONFIGS ------------------")
    tokenizer_name, context_length, dropout = validate_parameters(
        tokenizer_name, context_length, dropout, D_MODEL, NUM_HEADS
    )
    
    if tokenizer_name is None:
        return
    
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Context length: {context_length}")
    print(f"  Dropout: {dropout}")
    
    try:
        print(f"\n------------------ INITIALIZATION ------------------")
        token_embedder = TokenEmbedder(
            tokenizer_name=tokenizer_name,
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            context_length=context_length
        )
        
        mhsa_block = MultiHeadSelfAttention(
            d_model=D_MODEL,
            context_length=context_length,
            num_heads=NUM_HEADS,
            dropout=dropout
        )

        final_embeddings = token_embedder.process_text(input_text)
        output_vectors = mhsa_block(final_embeddings)
        
        print(f"\n------------------ FINAL RESULTS ------------------")
        print(f"Final output shape: {output_vectors.shape}")
        print(f"Context vectors computed for {context_length} positions")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main()