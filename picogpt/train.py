import math
import os
import torch
import numpy as np
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import argparse
from gpt import GPT, GPTConfig
from data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='PicoGPT Training')
    # Run configurations
    parser.add_argument('--baseline', action='store_true', help='Run baseline configuration (current code)')
    parser.add_argument('--run-1', action='store_true', help='Run configuration 1 (same as baseline)')
    parser.add_argument('--run-2', action='store_true', help='Run configuration 2 (Muon + RoPE + Data fixes)')
    # Individual feature flags
    parser.add_argument('--use-muon', action='store_true', help='Use Muon optimizer instead of AdamW')
    parser.add_argument('--use-rope', action='store_true', help='Use RoPE instead of learned positional embeddings')
    parser.add_argument('--data-fixes', action='store_true', help='Enable data shuffling and periodicity fixes')
    
    args = parser.parse_args()
    
    # Initialize defaults
    if not (args.baseline or args.run_1 or args.run_2 or args.use_muon or args.use_rope or args.data_fixes):
        # if no args, default to baseline
        args.use_muon = False
        args.use_rope = False
        args.data_fixes = False
    
    # if individual flags are set without run configs they remain as parsed
    # run configurations (these override individual flags)
    if args.run_1 or args.baseline:
        args.use_muon = False
        args.use_rope = False
        args.data_fixes = False
    elif args.run_2:
        args.use_muon = True
        args.use_rope = True
        args.data_fixes = True
        
    return args

def setup_ddp():
    #support distributed training with DDP
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        assert torch.cuda.is_available(), "need cuda to run ddp"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 #logging, checkpointing, etc
    else:
        #vanilla single gpu run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True #only gpu is the master
        #device detection -- should be cuda for my case
        device = "cpu" #good default
        if torch.cuda.is_available():
            device = "cuda" #work on my workstation
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps" # work on my macbook
        print(f"using device:", device)
        
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process

def log_model_stats(model, step, master_process):
    """Log model parameter statistics"""
    if master_process and step % 1000 == 0:  # Every 1000 steps
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get parameter norms
        param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/param_norm": param_norm,
            "step": step
        })

def get_adamw_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    #linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    if it > max_steps:
        return min_lr
    #in between the warmup and steps end, use cosine decaying down to the min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def main():
    args = parse_args()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()

    torch.manual_seed(1337) #reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    #gradient accumulation setup
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens to match gpt-3 paper
    B = 16   #micro batch size (64 on an 8gpu a100 training run)
    T = 1024 #sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure that the total batch size is dividisble by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B*T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoader(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process, shuffle=args.data_fixes)
    val_loader = DataLoader(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process, shuffle=False) #dont shuffle on validation

    torch.set_float32_matmul_precision("high") #TensorFloat32 avaliable on my 5060ti blackwell

    #create model
    # --- model = GPT.from_pretrained('gpt2') --- load og gpt2 weights but not necessary right now
    model = GPT(GPTConfig(vocab_size=50304, use_rope=args.use_rope)) #overriding the vocab size with a easier number for ops
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    max_lr = 6e-4 #gpt-3-small max lr
    min_lr = max_lr * 0.1
    warmup_steps = 715 #match gpt-3 warmup schedule
    max_steps = 19073 #match token shards

    if master_process:
        # Determine run name based on configuration
        if args.baseline or args.run_1:
            run_name = "baseline"
        elif args.run_2:
            run_name = "run-2-full"
        else:
            # Custom configuration
            features = []
            if args.use_muon: features.append("muon")
            if args.use_rope: features.append("rope")
            if args.data_fixes: features.append("datafixes")
            run_name = "-".join(features) if features else "baseline"
        
        wandb.init(
            entity="yarbrough-labs",
            project="PicoGPT",
            name=run_name,
            config={
                # Model hyperparameters
                "model_type": "gpt",
                "n_layer": raw_model.config.n_layer,
                "n_head": raw_model.config.n_head,
                "n_embd": raw_model.config.n_embd,
                "vocab_size": raw_model.config.vocab_size,
                "block_size": raw_model.config.block_size,
                
                # Training hyperparameters
                "batch_size": B,
                "sequence_length": T,
                "total_batch_size": total_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "weight_decay": 0.1,
                
                # Optimization config
                "use_muon": args.use_muon,
                "use_rope": args.use_rope,
                "data_fixes": args.data_fixes,
                "optimizer": "muon+adamw" if args.use_muon else "adamw",
                
                # Muon specific (if used)
                "muon_lr": 0.02 if args.use_muon else None,
                "muon_momentum": 0.95 if args.use_muon else None,
                
                # System
                "device": device,
                "ddp": ddp,
                "ddp_world_size": ddp_world_size if ddp else 1,
                "precision": "bfloat16",
            }
        )
        
        # Optionally watch model (can be heavy, comment out if too much)
        # wandb.watch(raw_model, log="all", log_freq=1000)
        
        print(f"\n✓ Weights & Biases initialized: {run_name}")
        print(f"  Project: PicoGPT")
        print(f"  Run URL: {wandb.run.get_url()}\n")


    #optimize
    if args.use_muon:
        optimizers = raw_model.configure_optimizers(
            weight_decay = 0.1,
            learning_rate=6e-4, #used in adamW component
            device=device,
            use_muon=True
        )
        muon_optimizer = optimizers['muon']
        adamw_optimizer = optimizers['adamw']

        if master_process:
            print("\n=== Using Muon + AdamW Hybrid ===")
            print("Muon: constant LR = 0.02")
            print("AdamW: scheduled LR starting at 6e-4")
            print("=" * 35 + "\n")
    else:
        optimizer = raw_model.configure_optimizers(
            weight_decay = 0.1,
            learning_rate=6e-4, 
            device=device,
            use_muon=False
        )

        if master_process:
            print("\n=== Using Pure AdamW ===")

    #training loop
    for step in range(max_steps):
        t0 = time.time()

        #validation loss split
        if step % 250 == 0: #every 250 steps validate
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision bf16 and tf32 
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                wandb.log({
                    "val/loss": val_loss_accum.item(),
                    "step": step
                })

        if step % 5000 == 0 or step == max_steps - 1:
            if master_process:
                if args.use_muon:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'muon_optimizer': muon_optimizer.state_dict(),
                        'adamw_optimizer': adamw_optimizer.state_dict(),
                        'step': step,
                        'config': raw_model.config
                    }
                else:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'config': raw_model.config
                    }
                torch.save(checkpoint, f"checkpoint_step_{step}.pt")
                wandb.save(f"checkpoint_step_{step}.pt")
                wandb.log({"checkpoint/step": step})

        model.train()

        if args.use_muon:
            muon_optimizer.zero_grad()
            adamw_optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss_accum = 0.0

        #gradient accumulation
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision bf16 and tf32 
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps #scaling down by the accum steps to recover the normalizer
            loss_accum += loss.detach()
            if ddp: #sync the last step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #average the loss accum in distrbuted processing
        
        #gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #cliping global norm as seen in gpt-3 (kinda hacky)
        
        #determine and set appropriate learning rate for this iter
        if args.use_muon:
            #muon uses a constant LR
            #only need to update adamw lr
            adamw_lr = get_adamw_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in adamw_optimizer.param_groups:
                param_group['lr'] = adamw_lr
        
            muon_optimizer.step()
            adamw_optimizer.step()
            lr = adamw_lr #logging
        else:
            lr = get_adamw_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000 #time difference in mill
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = (tokens_processed) / (dt / 1000) #seconds
        if master_process:
            print(f"step {step} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            
            # Log to wandb
            wandb.log({
                "train/loss": loss_accum.item(),
                "train/learning_rate": lr,
                "train/grad_norm": norm,
                "train/tokens_per_sec": tokens_per_sec,
                "train/step_time_ms": dt,
                "step": step
            })

            # Log model stats periodically
            log_model_stats(raw_model, step, master_process)
            
            # Log Muon LR separately if using Muon
            if args.use_muon:
                wandb.log({
                    "train/muon_lr": 0.02,
                    "train/adamw_lr": lr,
                    "step": step
                })
    if ddp: #cleanup
        destroy_process_group()

    if master_process:
        wandb.finish()
        print("\n✓ Training complete! Weights & Biases run finished.")

if __name__ == "__main__":
    main()
