# Four-Layer Neural Network

This directory contains the initial PyTorch scaffold for the Project 2 four-layer neural network.

In the course naming convention, a "4L" network means:

`input -> hidden layer 1 -> hidden layer 2 -> output`

The training script reads from `Project 2/Raw Datasets`:

- Auto MPG, target `mpg`
- House Price, target `House_Price`
- Parkinsons, target `total_UPDRS`
- Optional Parkinsons run for `motor_UPDRS`

Outputs are written to `outputs/`:

- `metrics_4l.csv`
- `best_configs_4l.csv`
- `plots/*.png`

## Usage

Install dependencies:

```bash
cd '/mnt/c/Users/legok/Documents/GitHub/Data-Science-II-Projects/Project 2/four-layer-nn'
pip install -r requirements.txt
```

Run all default datasets:

```bash
python3 train_4l_nn.py
```

Run Auto MPG only:

```bash
python3 train_4l_nn.py --dataset autompg
```

Run House Price only:

```bash
python3 train_4l_nn.py --dataset house
```

Run Parkinsons `total_UPDRS` only:

```bash
python3 train_4l_nn.py --dataset parkinsons_total
```

Run Parkinsons `motor_UPDRS` only:

```bash
python3 train_4l_nn.py --dataset parkinsons_motor
```

Run both Parkinsons targets:

```bash
python3 train_4l_nn.py --include-motor
```

If your instructor wants California Housing instead of the current house dataset, swap the house loader in `train_4l_nn.py` and keep the rest of the pipeline the same.
