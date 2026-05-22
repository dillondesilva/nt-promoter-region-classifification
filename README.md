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