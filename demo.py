# Load a model from experiment-with-genomics/glhp3oku/checkpoints/epoch=9-step=21310.ckpt

import torch
import tqdm
from model import AwesomeGenomicModel
from pipeline import LitModule
from transformers import AutoTokenizer
from datasets import load_dataset

model = LitModule.load_from_checkpoint("experiment-with-genomics/glhp3oku/checkpoints/epoch=9-step=21310.ckpt")
model = model.genomic_model
model.eval()

# Load a dataset from experiment-with-genomics/nucleotide_transformer_downstream_tasks/test.jsonl
ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")
ds = ds["test"].filter(lambda x: x["task"] == "promoter_all")

# Load a tokenizer from experiment-with-genomics/nucleotide_transformer_downstream_tasks/tokenizer.json
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")

for item in tqdm.tqdm(ds):
    input_ids = tokenizer.encode(item["sequence"], return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits, attn_weights = model(input_ids, attention_mask, return_attn_weights=True)

        # Assign attention weights to each token in the sequence
        decoded_tokens = tokenizer.decode(input_ids[0])
        decoded_tokens_list = decoded_tokens.split(" ")

        for i, token in enumerate(decoded_tokens_list):
            attn_weights_from_token = attn_weights[0, i, i]
            print(f"Token: {token}, Attention Weight: {attn_weights_from_token}")
