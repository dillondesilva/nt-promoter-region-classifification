import modal
import torch

app = modal.App("test-genomics-model-inference")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.8.0", "transformers>=4.44,<5", "accelerate", "fastapi[standard]", "pydantic", "lightning")
    .add_local_python_source("model", "pipeline")
)

volume = modal.Volume.from_name("test-nt-v3-example-weights")
MOUNT_DIR = "/experiment-with-genomics"
CKPT_PATH = "/experiment-with-genomics/experiment-with-genomics/glhp3oku/checkpoints/epoch=9-step=21310.ckpt"

@app.cls(
    gpu="T4",
    image=image,
    volumes={MOUNT_DIR: volume},
)
class GenomicsModelServing:
    @staticmethod
    def _instantiate_genomics_model():
        from pipeline import LitModule
        genomics_model = LitModule.load_from_checkpoint(CKPT_PATH)
        genomics_model = genomics_model.genomic_model
        genomics_model.eval()
        return genomics_model
    
    def _instantiate_tokenizer():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
        return tokenizer

    @modal.enter()
    def load(self):
        import torch
        from model import AwesomeGenomicModel
        from pipeline import LitModule
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        genomics_model = LitModule.load_from_checkpoint(CKPT_PATH)
        genomics_model = genomics_model.genomic_model
        genomics_model.eval()
        self.genomics_model = self._instantiate_genomics_model()
        self.tokenizer = self._instantiate_tokenizer()

    @modal.method()
    def run_inference(input_ids: torch.Tensor, attention_mask: torch.Tensor, return_attn_weights: bool = True) -> torch.Tensor:
        logits, attn_weights = self.genomics_model(input_ids, attention_mask, return_attn_weights=True)
        return logits, attn_weights
    
    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI
        from pydantic import BaseModel
        app = FastAPI()

        class RunInferenceRequest(BaseModel):
            sequence: str

        @app.post("/run_inference")
        def run_inference(request: RunInferenceRequest) -> torch.Tensor:
            input_ids = self.tokenizer.encode(request.sequence, return_tensors="pt")
            attention_mask = torch.ones_like(input_ids)
            logits, attn_weights = self.run_inference(input_ids, attention_mask, return_attn_weights=True)

            # Map the attention weights to each token
            decoded_tokens = self.tokenizer.decode(input_ids[0])
            decoded_tokens_list = decoded_tokens.split(" ")

            attn_weights_to_tokens = {}
            for i, token in enumerate(decoded_tokens_list):
                attn_weights_from_token = attn_weights[0, i, i].item()
                attn_weights_to_tokens[token] = attn_weights_from_token.item()

            return {
                "logits": logits.tolist(),
                "attn_weights_to_tokens": attn_weights_to_tokens,
            }

        return app
