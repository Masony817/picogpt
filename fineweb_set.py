import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset 
from tqdm import tqdm 

# ------------------------------------------
# config for the FineWeb dataset download and processing
LOCAL_DIRECTORY = "edu_fineweb10B"
DATASET_NAME = "sample-10BT"
TOKENS_PER_SHARD = int(1e8) # 100M tokens per shard

#local cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIRECTORY)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# load FineWeb-Edu dataset from HuggingFace
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=DATASET_NAME, split="train")


tokenizer = tiktoken.get_encoding("gpt2")
end_of_text_token = tokenizer._special_tokens['<|endoftext|>']

def tokenize_document(doc):
    """
    tokenizes a single document -  returns a numpy array of uint16 tokens.
    each document is prefixed with the <|endoftext|> token as a delimiter.
    """
    token_list = [end_of_text_token]
    token_list.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_array = np.array(token_list)
    assert (0 <= tokens_array).all() and (tokens_array < 2**16).all(), "token dictionary too large for uint16"
    return tokens_array.astype(np.uint16)

def save_shard_to_disk(filepath, token_array):
    np.save(filepath, token_array)

# process all documents and write shards
num_processes = max(1, os.cpu_count() // 2)
with mp.Pool(num_processes) as process_pool:
    current_shard_index = 0
    # pre-allocate buffer for the current shard
    shard_buffer = np.empty((TOKENS_PER_SHARD,), dtype=np.uint16)
    current_token_count = 0
    pbar = None
    
    for document_tokens in process_pool.imap(tokenize_document, dataset, chunksize=16):
        # check if current shard has enough space for new tokens
        if current_token_count + len(document_tokens) < TOKENS_PER_SHARD:
            # append tokens to current shard
            shard_buffer[current_token_count:current_token_count + len(document_tokens)] = document_tokens
            current_token_count += len(document_tokens)
            # init or update progress bar
            if pbar is None:
                pbar = tqdm(total=TOKENS_PER_SHARD, unit="tokens", desc=f"Shard {current_shard_index}")
            pbar.update(len(document_tokens))
        else:
            # current shard is full, write it and start a new one
            data_split = "val" if current_shard_index == 0 else "train"
            shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{data_split}_{current_shard_index:06d}")
            # fill remaining space in current shard
            tokens_to_fill = TOKENS_PER_SHARD - current_token_count
            pbar.update(tokens_to_fill)
            shard_buffer[current_token_count:current_token_count + tokens_to_fill] = document_tokens[:tokens_to_fill]
            save_shard_to_disk(shard_filename, shard_buffer)
            current_shard_index += 1
            pbar = None
            # start next shard with remaining tokens from current document
            leftover_tokens = document_tokens[tokens_to_fill:]
            shard_buffer[0:len(leftover_tokens)] = leftover_tokens
            current_token_count = len(leftover_tokens)

    # write final shard if there are remaining tokens
    if current_token_count != 0:
        data_split = "val" if current_shard_index == 0 else "train"
        shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{data_split}_{current_shard_index:06d}")
        save_shard_to_disk(shard_filename, shard_buffer[:current_token_count])