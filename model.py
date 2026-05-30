import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

class PromoterRegionPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class AwesomeGenomicModel(nn.Module):
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
        num_layers = self.backbone.config.num_hidden_layers + 1  # embedding + transformer layers
        self.weight_vector = nn.Parameter(torch.randn(num_layers).to(device))
        self.promoter_region_prediction_head = PromoterRegionPredictionHead(
            self.backbone.config.hidden_size,
            1,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_attn_weights: bool = False) -> torch.Tensor:
        # Move input_ids and attention_mask to the same device as the weight_vector
        input_ids = input_ids.to(self.weight_vector.device)
        attention_mask = attention_mask.to(self.weight_vector.device)
        with torch.no_grad():
            backbone_outs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attn_weights,
            )

        hidden_states = torch.stack(backbone_outs.hidden_states, dim=0)

        weights = torch.softmax(self.weight_vector, dim=0)
        weighted_hidden_states = (hidden_states * weights.view(-1, 1, 1, 1)).sum(dim=0)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (weighted_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if return_attn_weights:
            # note: use attentions from the most weighted layer only (workaround some OOMs)
            most_weighted_layer = weights.argmax()
            attn_weights_across_heads = backbone_outs.attentions[most_weighted_layer]
            pooled_attn_weights = attn_weights_across_heads.mean(dim=1)
            return self.promoter_region_prediction_head(pooled), pooled_attn_weights

        return self.promoter_region_prediction_head(pooled)
