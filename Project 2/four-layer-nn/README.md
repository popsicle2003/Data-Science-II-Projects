# Four-Layer Neural Network

This directory contains the initial PyTorch scaffold for the Project 2 four-layer neural network.

In the course naming convention, a "4L" network means:

`input -> hidden layer 1 -> hidden layer 2 -> output`

The training script reuses the Project 1 dataset conventions:

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
pip install -r requirements.txt
```

Run the default three-dataset workflow:

```bash
python3 train_4l_nn.py
```

Run one dataset only:

```bash
python3 train_4l_nn.py --dataset autompg
python3 train_4l_nn.py --dataset house
python3 train_4l_nn.py --dataset parkinsons_total
```

Include the extra Parkinsons `motor_UPDRS` target:

```bash
python3 train_4l_nn.py --include-motor
```

If your instructor wants California Housing instead of the Project 1 house dataset, swap the house loader in `train_4l_nn.py` and keep the rest of the pipeline the same.
