class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
    
    @staticmethod
    def _get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    @staticmethod
    def _merge(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >=  256
        num_merges = vocab_size - 256
        
        ids = list(text.encode("utf-8"))
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            
            if verbose:
                print(f"Merging {top_pair} into new token {idx}")
                
            ids = self._merge(ids, top_pair, idx)
            self.merges[top_pair] = idx

        self._build_vocab()
        
    def _build_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break
                
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
            
        return tokens
    
