# Playing around with Nucleotide Transformer

This repo contains the code for a recent article I wrote. In it, we train a linear probe on top of the Nucleotide Transformer model to predict promoter regions.

In this repository, my goal is to clearly demonstrate how the backbone can be applied to solve the binary classification problem of predicting if a given genome sequence (of length 300 base pairs) is a promoter region. To evaluate how well we can classify such sequences using the Nucleotide Transformer, we will use a linear probe to learn a direct mapping from the backbone outputs to a binary class.

My recommendation is to read this article section as a primer for what is contained within the codebase, followed by then working through the repo contents. I have tried to keep it lightweight and simple :)

The key files within the repository are as follows:

```
cli.py - Comes with a CLI script you can use to run the train.

pipeline.py - Comes with a Pytorch Lightning module that can be run to train the model/

model.py - This is probably the key file to view. Inside the forward pass, you will notice how the weight vector has been implemented as well as the method used to obtain attention weights which are visualised in the demo.
```

# Running the train

1. Ensure you have uv installed (see `https://docs.astral.sh/uv/getting-started/installation/#installation-methods`)

2. Create a new venv:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

3. Install the dependencies:

```bash
uv sync
```

4. Run the train (half-sized):

```
python cli.py --train --dataset-size 0.5
```