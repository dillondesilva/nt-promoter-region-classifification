import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.callbacks import EarlyStopping
from pipeline import LitModule
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import modal

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
MAX_LENGTH = 512
DATASET_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
TASK_NAME = "promoter_all"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-size", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of task train rows used for training (remainder is validation)",
    )
    return parser.parse_args()

def generate_train_val_split(ds, train_fraction: float, seed: int):
    split = ds.train_test_split(train_size=train_fraction, seed=seed)
    return split["train"], split["test"]

def run_train(args):
    model = LitModule()
    wandb_logger = WandbLogger(project="experiment-with-genomics")
    ds = load_dataset(DATASET_NAME)
    task_train = ds["train"].filter(lambda x: x["task"] == TASK_NAME)
    task_test = ds["test"].filter(lambda x: x["task"] == TASK_NAME)
    
    # Apply fraction to task_train
    task_train = task_train.select(range(int(len(task_train) * args.dataset_size)))

    train_ds, val_ds = generate_train_val_split(
        task_train, args.train_fraction, args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    test_loader = DataLoader(
        task_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    callbacks = [EarlyStopping(monitor="val_loss", patience=3)]
    trainer = L.Trainer(max_epochs=args.epochs, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        run_train(args)
