import torch
import wandb
import numpy as np
import pandas as pd
import json
import gc
import os
from collections import defaultdict
from datasets import load_dataset

# ensure lm-eval is installed
try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("error: lm_eval not installed.")
    print("please run: pip install -r requirements.txt")
    exit(1)

from checkpoint_manager import load_model

# init wandb
wandb.init(project="PicoGPT-Evals", name="comprehensive-comparison")

# eval models
models_config = {
    'gpt2-base': 'gpt2',
    'run1-3.44loss': 'baseline',
    'run2-3.74loss': 'run-2',
}

# config
## note: batch_size must be 1 because PicoGPT doesn't support attention_mask for padding yet ##
BATCH_SIZE = 1 
LM_EVAL_TASKS = ['hellaswag', 'piqa', 'winogrande', 'arc_easy', 'boolq']

GENERATION_PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "The American Civil War began in",
    "If you drop a ball from a building, it will",
    "To bake a cake, first you need to",
    "Once upon a time, there was a",
    "The detective walked into the room and saw",
    "In Python, you can define a function using",
]

# final aggregation store
all_results = {
    'benchmarks': {},
    'perplexity': defaultdict(dict),
    'generations': defaultdict(dict),
    'diversity': {}
}

def calculate_perplexity(model, tokenizer, dataset, max_length=1024, limit=100):
    """
    Calculates perplexity correctly by summing NLL over all tokens.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    # handle different dataset types
    if hasattr(dataset, 'take'): # streaming
        subset = dataset.take(limit)
    elif hasattr(dataset, 'select'): # hf dataset
        subset = dataset.select(range(min(len(dataset), limit)))
    else: # list
        subset = dataset[:limit]
        if isinstance(subset, dict): # dict of lists
            subset = [dict(zip(subset, t)) for t in zip(*subset.values())]

    with torch.no_grad():
        for i, item in enumerate(subset):
            text = item.get('text', '') or item.get('content', '') or item.get('code', '')
            if not text or len(text.strip()) < 10:
                continue
            
            encodings = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
            input_ids = encodings.input_ids.to(model.device)
            
            if input_ids.size(1) < 2:
                continue
                
            target_ids = input_ids.clone()
            
            # forward pass
            outputs = model(input_ids, labels=target_ids)
            
            # recover total NLL for this sequence (mean_loss * seq_len)
            # the loss is calculated on seq_len - 1 tokens (next token prediction)
            seq_len = input_ids.size(1)
            pred_len = seq_len - 1
            if pred_len > 0:
                total_nll += outputs.loss.item() * pred_len
                total_tokens += pred_len
            
    if total_tokens == 0:
        return float('inf')
        
    return np.exp(total_nll / total_tokens)

def measure_repetition(text, n=3):
    """Calculate n-gram diversity (higher = less repetitive)"""
    tokens = text.split()
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def verify_environment(models_config, eval_datasets):
    print("\n" + "="*60)
    print("pre-flight check")
    print("="*60)
    
    # check datasets
    print("checking datasets...")
    if not eval_datasets:
        print(" no datasets loaded!")
    
    for name, ds in eval_datasets.items():
        try:
            if hasattr(ds, 'take'):
                next(iter(ds.take(1)))
            elif hasattr(ds, 'select'):
                _ = ds.select(range(1))[0]
            else:
                _ = ds[0]
            print(f" -> dataset '{name}' accessible")
        except Exception as e:
            raise RuntimeError(f"dataset '{name}' check failed: {e}")

    # load/check models
    print("\nchecking models...")
    for name, model_id in models_config.items():
        print(f" -> verifying {name}...")
        try:
            model, tokenizer = load_model(model_id)
            
            # quick forward pass to verify
            inp = tokenizer("test", return_tensors='pt').to(model.device)
            with torch.no_grad():
                model(inp.input_ids)
                
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f" -> model '{name}' loadable and runnable")
        except Exception as e:
            raise RuntimeError(f"model '{name}' check failed: {e}")

    print("\npre-flight checks passed.")
    print("="*60 + "\n")


#eval loop
print("\n" + "="*60)
print("starting eval...")
print("="*60 + "\n")

# pre-load datasets
print("loading datasets...")
try:
    # wikitext for ppl
    wiki_data = load_dataset('wikitext', 'wikitext-2-v1', split='test') 
    code_data = load_dataset('codeparrot/github-code', languages=['Python'], split='train', streaming=True)
    book_data = load_dataset('bookcorpus', split='train', streaming=True)
    
    eval_datasets = {
        'wikitext': wiki_data,
        'code': code_data,
        'books': book_data
    }
except Exception as e:
    print(f"error loading datasets: {e}")
    eval_datasets = {}

# Verify environment before starting
try:
    verify_environment(models_config, eval_datasets)
except Exception as e:
    print(f"\npre-flight check failed: {e}")
    print("aborting evaluation!")
    exit(1)

for name, model_id in models_config.items():
    print(f"\n" + "-"*40)
    print(f"processing Model: {name} ({model_id})")
    print("-" * 40)
    
    try:
        #load model
        model, tokenizer = load_model(model_id)
        device = model.device
        
        #eval harness
        print(f" -> running lm-eval benchmarks ({', '.join(LM_EVAL_TASKS)})...")
        try:
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)
            
            results = simple_evaluate(
                model=lm,
                tasks=LM_EVAL_TASKS,
                num_fewshot=0,
                batch_size=BATCH_SIZE,
                device=str(device)
            )
            
            model_scores = {}
            for task in LM_EVAL_TASKS:
                if 'results' in results and task in results['results']:
                    # try acc,none first, fallback to acc
                    acc = results['results'][task].get('acc,none', results['results'][task].get('acc', 0.0))
                    model_scores[task] = acc
                    wandb.log({f"{name}/{task}": acc})
            
            all_results['benchmarks'][name] = model_scores
            print("    done.")
            
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

        #perplexity test
        print(" -> calculating Perplexity...")
        for domain, dataset in eval_datasets.items():
            try:
                ppl = calculate_perplexity(model, tokenizer, dataset, limit=200)
                all_results['perplexity'][name][domain] = ppl
                wandb.log({f"{name}/{domain}_ppl": ppl})
                print(f"    {domain}: {ppl:.2f}")
            except Exception as e:
                print(f"    {domain} FAILED: {e}")

        #generation and diversity
        print("  -> generating samples...")
        diversity_scores = []
        
        for prompt in GENERATION_PROMPTS:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Store generation
            if prompt not in all_results['generations']:
                all_results['generations'][prompt] = {}
            all_results['generations'][prompt][name] = generated_text
            
            # Calculate diversity
            diversity_scores.append(measure_repetition(generated_text))
            
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        all_results['diversity'][name] = avg_diversity
        wandb.log({f"{name}/diversity": avg_diversity})
        print(f"    avg diversity: {avg_diversity:.4f}")

        # CLEANUP
        del model
        del tokenizer
        if 'lm' in locals(): del lm
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"! error processing {name}: {e}")
        continue

#reporting
print("\n" + "="*60)
print("getting final results...")
print("="*60)

# benchmark
if all_results['benchmarks']:
    df_bench = pd.DataFrame(all_results['benchmarks']).T
    print("\nbenchmarks:")
    print(df_bench)
    wandb.log({"benchmark_summary": wandb.Table(dataframe=df_bench.reset_index())})

# perplexity
if all_results['perplexity']:
    df_ppl = pd.DataFrame(all_results['perplexity']).T
    print("\nperplexity:")
    print(df_ppl)
    wandb.log({"perplexity_summary": wandb.Table(dataframe=df_ppl.reset_index())})

#generations
gen_rows = []
for prompt, models_res in all_results['generations'].items():
    row = {'prompt': prompt}
    row.update(models_res)
    gen_rows.append(row)

if gen_rows:
    df_gen = pd.DataFrame(gen_rows)
    wandb.log({"generations_comparison": wandb.Table(dataframe=df_gen)})

# local json
with open('eval_results.json', 'w') as f:
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
    json.dump(all_results, f, indent=2, default=convert)

print(f"\nresults saved to eval_results.json")
print(f"WandB run: {wandb.run.url}")
