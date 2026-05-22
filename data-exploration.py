from datasets import load_dataset

DATASET_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
TASK_NAME = "promoter_all"

ds = load_dataset(DATASET_NAME)
train_ds = ds["train"].filter(lambda x: x["task"] == TASK_NAME)
test_ds = ds["test"].filter(lambda x: x["task"] == TASK_NAME)

# Number of items in the train set
print("Number of items in the train set:")
print(len(train_ds))

# Example of item in the train set
print("Example of item in the train set:")
print(train_ds[0])
