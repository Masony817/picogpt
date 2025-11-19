import os
import torch
import wandb
import glob
import re
from typing import Optional, Tuple, Union, List
from dataclasses import asdict

from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

#local model definitions
from gpt import GPT, GPTConfig

class PicoGPTHFConfig(PretrainedConfig):
    "HuggingFace-compatable config wrapper for PicoGPT"

    model_type = 'picogpt'

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        use_rope=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.use_rope = use_rope
        super().__init__(**kwargs)

    @classmethod
    def from_gpt_config(cls, config: GPTConfig):
        return cls(
            vocab_size=config.vocab_size,
            n_positions=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            use_rope=config.use_rope,
        )

class PicoGPTHF(PreTrainedModel):

    config_class = PicoGPTHFConfig

    def __init__(self, config: PicoGPTHFConfig, model: Optional[GPT] = None):
        super().__init__(config)
        if model is None:
            #no model instance provided, init a fresh one (mostly for HF internal init)
            gpt_config = GPTConfig(
                vocab_size=config.vocab_size,
                block_size=config.n_positions,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                use_rope=config.use_rope
            )
            self.model = GPT(gpt_config)
        else:
            self.model = model

        self.main_input_name = "input_ids" #should figure out what this is tbh 

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        # PicoGPT doesn't natively use the attention_mask and it assumes causal masking only,
        # but we accept it to conform to HF API.
        
        # forward pass in the underlying model
        # GPT.forward returns (logits, loss)
        logits, loss = self.model(input_ids, targets=labels)

        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # minimal implementation for generation support
        return {"input_ids": input_ids}

    def _reorder_cache(self, past, beam_idx):
        # required for beam search, but PicoGPT doesn't support kv-caching yet
        raise NotImplementedError("PicoGPT does not support beam search caching yet.")

def get_wandb_checkpoint(run_name_query: str, project="PicoGPT", entity="yarbrough-labs", download_root='checkpoints') -> str:

    api = wandb.api()

    runs = api.runs(f"{entity}/{project}")
    matched_runs = [r for r in runs if run_name_query in r.name or run_name_query == r.name]

    if not matched_runs:
        #fallback - explicit name matching from the @train.py
        if run_name_query == 'run-2' or run_name_query == 'baseline':
            matched_runs = [r for r in runs if 'run-2' in r.name or 'baseline' in r.name]
    
    if not matched_runs:
        raise ValueError(f"no runs found matching'{run_name_query}' in {entity}/{project}")
    
    target_run = matched_runs[0]
    print(f"found run: {target_run.name} (ID: {target_run.id})")

    files = target_run.files()
    checkpoint_files = [f for f in files if f.name.startswith("checkpoint_step_") and f.name.endswith(".pt")]

    if not checkpoint_files:
        raise FileNotFoundError(f"no checkpoints found for run {target_run.name}")
    
    def get_step(filename):
        match = re.search(r"checkpoint_step_(/d+)\.pt", filename)
        return int(match.group(1) if match else -1)
    
    latest_ckpt = max(checkpoint_files, key=lambda x: get_step(x.name))
    
    local_dir = os.path.join(download_root, target_run.name)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, latest_ckpt.name)

    if not os.path.exists(local_path):
        print(f"downloading {latest_ckpt.name}...")
        latest_ckpt.download(root=local_dir, replace=True)
    else:
        print(f"using cached checkpoint: {local_path}")
    
    return local_path

def load_model(model_identifier: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads a model for inference/eval.
    
    Args:
        model_identifier: "baseline", "run-2", or "gpt2" (or any HF model ID)
        device: "cuda", "cpu", or "mps"
    
    Returns: model (PreTrainedModel), tokenizer
    """

    print(f"loading model: {model_identifier} on {device}")

    #standard hugging face models
    if model_identifier == 'gpt2' or '/' in model_identifier:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_identifier)
            tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            model.to(device)
            model.eval()
            return model, tokenizer
        except OSError:
            #not hf model just pass to wandb check
            pass
    
    #custom wandb runs
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        checkpoint_path = get_wandb_checkpoint(model_identifier)
    except Exception as e:
        print(f"couldnt get download from WandB: {e}")
        print(f"searching local filesystem...")
        #fallback to check local dir
        if os.path.exists(model_identifier):
            checkpoint_path = model_identifier
        else:
            raise ValueError(f"Couldn't find model/run '{model_identifier} locally or on WanB")
    
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config and state dict
    if 'config' in checkpoint:
        gpt_conf = checkpoint['config']
    else:
        gpt_conf = GPTConfig() # Default
        print("Warning: No config found in checkpoint, using default GPTConfig")
    
    state_dict = checkpoint['model']
    
    # clean up state_dict (remove DDP prefixes if present)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    #init base model
    base_model = GPT(gpt_conf)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    base_model.eval()
    
    # hf compatability wrap
    hf_config = PicoGPTHFConfig.from_gpt_config(gpt_conf)
    wrapped_model = PicoGPTHF(hf_config, model=base_model)
    wrapped_model.to(device)
    
    return wrapped_model, tokenizer