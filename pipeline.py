"""Minimal PyTorch Lightning training / validation scaffold."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import AwesomeGenomicModel
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
    binary_precision,
    binary_recall,
    binary_specificity,
)
from transformers import AutoTokenizer

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Binary classification metrics from logits and integer class labels."""
    labels = labels.int()
    probs = F.softmax(logits, dim=-1)[:, 1]
    preds = logits.argmax(dim=-1)
    return {
        "auc_roc": binary_auroc(probs, labels),
        "accuracy": binary_accuracy(preds, labels),
        "sensitivity": binary_recall(preds, labels),
        "specificity": binary_specificity(preds, labels),
        "precision": binary_precision(preds, labels),
        "recall": binary_recall(preds, labels),
        "f1": binary_f1_score(preds, labels),
    }


def _preds_probs_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels.int()
    probs = F.softmax(logits, dim=-1)[:, 1]
    preds = logits.argmax(dim=-1)
    return preds, probs, labels


def build_classification_metrics() -> MetricCollection:
    """Stateful metrics for Lightning; accumulates correctly across batches and ranks."""
    return MetricCollection(
        {
            "auc_roc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "sensitivity": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
        }
    )


def update_classification_metrics(
    metrics: MetricCollection,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    preds, probs, labels = _preds_probs_labels(logits, labels)
    metrics["auc_roc"].update(probs, labels)
    metrics["accuracy"].update(preds, labels)
    metrics["sensitivity"].update(preds, labels)
    metrics["specificity"].update(preds, labels)
    metrics["precision"].update(preds, labels)
    metrics["recall"].update(preds, labels)
    metrics["f1"].update(preds, labels)


class LitModule(L.LightningModule):
    """Replace `self.net` and batch unpacking with your model and data."""

    def __init__(self, num_classes: int = 2, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.genomic_model = AwesomeGenomicModel()
        self.train_metrics = build_classification_metrics()
        self.val_metrics = build_classification_metrics()
        self.test_metrics = build_classification_metrics()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.genomic_model(x)

    def collate_batch(
        self,
        batch: dict[str, list] | list[dict[str, object] | tuple[str, int]],
        max_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            sequences = list(batch["sequence"])
            labels = torch.tensor(batch["label"], dtype=torch.long)
        elif isinstance(batch[0], dict):
            sequences = [item["sequence"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        else:
            sequences = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        enc = self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        return input_ids, attention_mask, labels

    def _metrics_for_stage(self, stage: str) -> MetricCollection:
        return {"train": self.train_metrics, "val": self.val_metrics, "test": self.test_metrics}[stage]

    def _log_stage_metrics(self, stage: str, batch_size: int) -> None:
        prog_bar_metrics = {"accuracy", "auc_roc", "f1"}
        for name, metric in self._metrics_for_stage(stage).items():
            self.log(
                f"{stage}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=name in prog_bar_metrics,
                batch_size=batch_size,
            )

    def _classification_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        input_ids, attention_mask, labels = self.collate_batch(batch)
        logits = self.genomic_model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        batch_size = input_ids.size(0)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch_size,
        )
        update_classification_metrics(self._metrics_for_stage(stage), logits, labels)
        self._log_stage_metrics(stage, batch_size)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._classification_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._classification_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._classification_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.genomic_model.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    model = LitModule()
    print(model)