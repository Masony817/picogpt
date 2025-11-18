import os
import numpy as np
import torch

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    #overly simple dataloader with distributed loading if needed
    def __init__(self, B, T, process_rank, num_processes, split, master_process=False, data_root='edu_fineweb10B', seed=42, shuffle=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seed=seed
        self.shuffle=shuffle
        assert split in {'train', 'val'}

        #get shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        if self.shuffle:
            self.rng = np.random.RandomState(seed)
            self.shard_indices = list(range(len(self.shards)))
            self.rng.shuffle(self.shard_indices)
        else:
            self.shard_indices = list(range(len(self.shards)))

        #state, init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.shard_indices[self.current_shard]])
        self.current_pos = self.B * self.T * self.process_rank

    
    def reset(self):
        self.current_shard = 0
        if self.shuffle:
            self.rng.shuffle(self.shard_indices)
        self.tokens = load_tokens(self.shards[self.shard_indices[self.current_shard]])
        self.current_pos = self.B * self.T * self.process_rank

    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos: self.current_pos+B*T+1]
        x = buf[:-1].view(B, T) #inputs
        y = buf[1:].view(B, T) #targets
        #advance tensor pos
        self.current_pos += B * T * self.num_processes
        #reset if loading the next batch would be out of dataset bounds
        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            #reshuffle at epoch completion
            if self.shuffle and self.current_shard == 0:
                self.rng.shuffle(self.shard_indices)
            self.tokens = load_tokens(self.shards[self.shard_indices[self.current_shard]])
            self.current_pos = self.B * self.T * self.process_rank
        return x, y

