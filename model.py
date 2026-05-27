import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

class PromoterRegionPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, mlp_hidden_dim: int = 192):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class AwesomeGenomicModel(nn.Module):
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
        num_layers = self.backbone.config.num_hidden_layers + 1  # embedding + transformer layers
        self.weight_vector = nn.Parameter(torch.randn(num_layers).to(device))
        self.promoter_region_prediction_head = PromoterRegionPredictionHead(
            self.backbone.config.hidden_size,
            2,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Move input_ids and attention_mask to the same device as the weight_vector
        input_ids = input_ids.to(self.weight_vector.device)
        attention_mask = attention_mask.to(self.weight_vector.device)
        with torch.no_grad():
            backbone_outs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = torch.stack(backbone_outs.hidden_states, dim=0)

        weights = torch.softmax(self.weight_vector, dim=0)
        weighted_hidden_states = (hidden_states * weights.view(-1, 1, 1, 1)).sum(dim=0)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (weighted_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.promoter_region_prediction_head(pooled)
