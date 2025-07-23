import os
import glob
import torch
import tiktoken
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    """
    Dataset class for loading and tokenizing text files for LLM training.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_length: int = 512,
        stride: int = 256,
        tokenizer_name: str = "gpt2"
    ):
        """
        Initialize the Dataset

        Args:
            data_dir (str): Directory containing text files
            max_length (int, optional): Maximum sequence length (context window). Defaults to 512.
            stride (int, optional): Step size for sliding window (controls overlap). Defaults to 256.
            tokenizer_name (str, optional): Tokenizer to use (gpt2, gpt-3.5-turbo, etc...). Defaults to "gpt2".
        """
        
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.stride = stride
        
        ## Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception as e:
            logger.error(f"No tokenizer named {tokenizer_name} found!")
            exit()
        
        ## Find all text files
        self.text_files = self._find_text_files()
        logger.info(f"Found {len(self.text_files)} text files")
        
        ## Pre-compute all chunks
        self.chunks = self._prepare_chunks()
        logger.info(f"Created {len(self.chunks)} training chunks")
        
    def _find_text_files(self) -> List[Path]:
        """
        Finds all text files in a given directory

        Returns:
            List[Path]: List of all paths of the text files
        """
        ## Extensions supported
        extensions = ['*.txt', '*.md']
        files = []
        
        ## Add files with the supported extensions to the files list
        for ext in extensions:
            files.extend(self.data_dir.glob(f"**/{ext}"))
            
        return sorted(files)
    
    def _prepare_chunks(self) -> List[Tuple[str, int, int]]:
        """
        Pre-process all files and create chunks.

        Returns:
            List[Tuple[str, int, int]]: Returns list of (file_path, start_idx, end_idx) tuples.
        """
        
        chunks = []
        
        for file_path in self.text_files:
            try:
                ## Try to read and tokenize the file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                if not text.strip():
                    continue
                
                token_ids = self.tokenizer.encode(text)
                
                ## Create sliding window chunks
                for i in range(0, len(token_ids)- self.max_length, self.stride):
                    chunks.append((str(file_path), i, i + self.max_length))
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
            
        return chunks
    
    def __len__(self) -> int:
        """
        Returns total number of training samples
        """
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            input_ids: Input token sequence of shape [max_length]
            target_ids: Target token sequence of shape [max_length] (shifted by 1)
        """
        file_path, start_idx, end_idx = self.chunks[idx]
        
        ## Read and tokenize the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        token_ids = self.tokenizer.encode(text)
        
        ## Extract the chunk
        input_chunk = token_ids[start_idx:end_idx]
        target_chunk = token_ids[start_idx + 1:end_idx + 1]
        
        ## Convert to tensors
        input_ids = torch.tensor(input_chunk, dtype=torch.long)
        target_ids = torch.tensor(target_chunk, dtype=torch.long)
        
        return input_ids, target_ids
    
def create_llm_dataloader(
    data_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
    stride: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer_name: str = "gpt2"
) -> DataLoader:
    """
    Create a DataLoader for LLM training.
    
    Args:
        data_dir: Directory containing text files
        batch_size: Batch size for training
        max_length: Maximum sequence length (context window)
        stride: Step size for sliding window
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        tokenizer_name: Tokenizer to use
    
    Returns:
        DataLoader ready for training
    """
    dataset = MyDataset(
        data_dir=data_dir,
        max_length=max_length,
        stride=stride,
        tokenizer_name=tokenizer_name
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=True 
    )
    
    return dataloader

def analyze_dataset(
    data_dir: str, 
    tokenizer_name: str = "gpt2"
    ) -> dict:
    """
    Analyze the dataset to provide useful statistics.
    
    Returns:
        Dictionary with dataset statistics
    """
    
    logger.info("Analyzing Dataset...")
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    ## Find all text files
    data_dir = Path(data_dir)
    extensions = ['*.txt', '*.md']
    files = []
    
    for ext in extensions:
        files.extend(data_dir.glob(f"**/{ext}"))
        
    total_chars = 0
    total_tokens = 0
    file_count = len(files)
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                
            total_chars += len(text)
            total_tokens += len(tokenizer.encode(text))
            
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            file_count -= 1
        
    vocab_size = tokenizer.n_vocab
    
    stats = {
        'total_files': file_count,
        'total_characters': total_chars,
        'total_tokens': total_tokens,
        'vocab_size': vocab_size,
        'avg_chars_per_file': total_chars / file_count if file_count > 0 else 0,
        'avg_tokens_per_file': total_tokens / file_count if file_count > 0 else 0,
    }
    
    return stats

def main():
    """
    Main Function to run the dataloader
    """
    
    parser = argparse.ArgumentParser(description="LLM Training DataLoader")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing text files")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256,
                       help="Stride for sliding window")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze dataset statistics")
    
    args = parser.parse_args()
    
    ## Check if the directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory {args.data_dir} does not exist!")
        return
    
    ## Analyze the dataset
    if args.analyze:
        stats = analyze_dataset(args.data_dir)
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key:25}: {value:,}")
        print("="*50)
        
    ## Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_llm_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride
    )
    
    logger.info(f"Dataloader ready!")
    logger.info(f"Total samples: {len(dataloader.dataset):,}")
    logger.info(f"Total batches: {len(dataloader):,}")

if __name__ == "__main__":
    main()