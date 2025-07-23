import regex as re

class RegexTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
    
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
        
        text_chunks = re.findall(self.compiled_pattern, text)
        
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges):
            stats = {}
            
            for chunk_ids in ids:
                chunk_stats = self._get_stats(chunk_ids)
                for pair, count in chunk_stats.items():
                    stats[pair] = stats.get(pair, 0) + count
            
            if not stats:
                break        
            
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            
            if verbose:
                print(f"Merging {top_pair} -> {idx}")

            new_ids = []
            for chunk_ids in ids:
                new_ids.append(self._merge(chunk_ids, top_pair, idx))
            ids = new_ids
            
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
        final_ids = []
        text_chunks = re.findall(self.compiled_pattern, text)
        
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            
            while len(chunk_ids) >= 2:
                stats = self._get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                
                if pair not in self.merges:
                    break
                
                idx = self.merges[pair]
                chunk_ids = self._merge(chunk_ids, pair, idx)
            
            final_ids.extend(chunk_ids)
            
        return final_ids