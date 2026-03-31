#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

PROJECT2_DATA_DIR = REPO_ROOT / "Project 2" / "Raw Datasets"

AUTO_MPG_PATH = PROJECT2_DATA_DIR / "auto-mpg" / "auto-mpg.data-original"
HOUSE_PATH = PROJECT2_DATA_DIR / "house_price_regression_dataset" / "house_price_regression_dataset.csv"
PARK_PATH_CANDIDATES = (
    PROJECT2_DATA_DIR / "parkinsons_telemonitoring" / "parkinsons_updrs.data",
    PROJECT2_DATA_DIR / "parkinsons+telemonitoring" / "parkinsons_updrs.data",
)

OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

DEFAULT_HIDDEN_PAIRS = ((16, 8), (32, 16), (64, 32))
DEFAULT_ACTIVATIONS = ("relu", "tanh", "leaky_relu")
DEFAULT_BATCH_SIZES = (16, 32)
DEFAULT_LEARNING_RATES = (1e-3, 5e-4)


@dataclass
class DatasetSpec:
    slug: str
    display_name: str
    target: str
    X: pd.DataFrame
    y: pd.Series


@dataclass
class PreparedData:
    spec: DatasetSpec
    X_train: np.ndarray
    X_val: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_scaler: StandardScaler


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_outputs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def first_existing_path(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find any of these dataset files:\n{tried}")


def parse_auto_mpg(path: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r'^\s*([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+"(.*)"\s*$'
    )

    with path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                warnings.warn(f"Skipping unparsable Auto MPG row: {line[:80]}")
                continue

            mpg, cyl, disp, hp, wt, acc, year, origin, name = match.groups()
            rows.append(
                {
                    "mpg": mpg,
                    "cylinders": cyl,
                    "displacement": disp,
                    "horsepower": hp,
                    "weight": wt,
                    "acceleration": acc,
                    "model_year": year,
                    "origin": origin,
                    "car_name": name,
                }
            )

    df = pd.DataFrame(rows)
    for column in ("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"):
        df[column] = pd.to_numeric(df[column].replace({"?": np.nan, "NA": np.nan}), errors="coerce")
    return df


def preprocess_auto_mpg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=["mpg"]).copy()
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median(skipna=True))
    df = df.drop(columns=["car_name"])

    df["origin"] = df["origin"].astype(int)
    origin_oh = pd.get_dummies(df["origin"], prefix="origin", drop_first=False)
    df = pd.concat([df.drop(columns=["origin"]), origin_oh], axis=1)

    y = df["mpg"].astype(float)
    X = df.drop(columns=["mpg"]).astype(float)
    return X, y


def preprocess_house(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["House_Price"].astype(float)
    X = df.drop(columns=["House_Price"]).astype(float)
    return X, y


def preprocess_parkinsons(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[target].astype(float)
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")

    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep].astype(float)
    y = y.loc[keep].astype(float)
    return X, y


def load_dataset_specs(include_motor: bool) -> dict[str, DatasetSpec]:
    auto_raw = parse_auto_mpg(AUTO_MPG_PATH)
    X_auto, y_auto = preprocess_auto_mpg(auto_raw)

    X_house, y_house = preprocess_house(HOUSE_PATH)
    park_path = first_existing_path(PARK_PATH_CANDIDATES)
    X_total, y_total = preprocess_parkinsons(park_path, target="total_UPDRS")

    specs = {
        "autompg": DatasetSpec("autompg", "Auto MPG", "mpg", X_auto, y_auto),
        "house": DatasetSpec("house", "House Price", "House_Price", X_house, y_house),
        "parkinsons_total": DatasetSpec("parkinsons_total", "Parkinsons", "total_UPDRS", X_total, y_total),
    }

    if include_motor:
        X_motor, y_motor = preprocess_parkinsons(park_path, target="motor_UPDRS")
        specs["parkinsons_motor"] = DatasetSpec("parkinsons_motor", "Parkinsons", "motor_UPDRS", X_motor, y_motor)

    return specs


def prepare_dataset(spec: DatasetSpec) -> PreparedData:
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        spec.X.values,
        spec.y.values,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()

    return PreparedData(
        spec=spec,
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        y_scaler=y_scaler,
    )


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            cuda_runtime = torch.version.cuda or "none"
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                f"Current PyTorch CUDA runtime: {cuda_runtime}. "
                "Install a CUDA-enabled PyTorch build, then rerun with --device cuda."
            )
        return torch.device("cuda")
    if name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested, but it is not available in this PyTorch build.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_status(requested_name: str, resolved_device: torch.device) -> None:
    print(f"Requested device: {requested_name}")
    print(f"Using device: {resolved_device}")
    print(f"PyTorch version: {torch.__version__}")

    if resolved_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: True")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {gpu_name}")
    elif resolved_device.type == "mps":
        print("MPS available: True")
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA runtime: {torch.version.cuda or 'none'}")
        if not torch.cuda.is_available():
            print("GPU note: PyTorch is currently falling back to CPU. If you expected CUDA, install a CUDA-enabled PyTorch build and run with --device cuda.")


def configure_runtime(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


class FourLayerRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, activation_name: str) -> None:
        super().__init__()
        activation_factory = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }[activation_name]

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            activation_factory(),
            nn.Linear(hidden1, hidden2),
            activation_factory(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def inverse_targets(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def save_pred_vs_actual_plot(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=16, alpha=0.7)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([low, high], [low, high], linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def train_one_config(
    prepared: PreparedData,
    hidden_pair: tuple[int, int],
    activation_name: str,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    set_seed()

    model = FourLayerRegressor(
        input_dim=prepared.X_train.shape[1],
        hidden1=hidden_pair[0],
        hidden2=hidden_pair[1],
        activation_name=activation_name,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    use_cuda_loader = device.type == "cuda"

    X_train_tensor = torch.tensor(prepared.X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(prepared.y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(prepared.X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(prepared.y_val_scaled, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        pin_memory=use_cuda_loader,
    )

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb_cpu, yb_cpu in train_loader:
            xb = xb_cpu.to(device, non_blocking=use_cuda_loader)
            yb = yb_cpu.to(device, non_blocking=use_cuda_loader)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = loss_fn(val_preds, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_scaled_pred = model(torch.tensor(prepared.X_train, dtype=torch.float32, device=device)).cpu().numpy()
        val_scaled_pred = model(torch.tensor(prepared.X_val, dtype=torch.float32, device=device)).cpu().numpy()

    y_train_pred = inverse_targets(train_scaled_pred, prepared.y_scaler)
    y_val_pred = inverse_targets(val_scaled_pred, prepared.y_scaler)

    return {
        "hidden1": hidden_pair[0],
        "hidden2": hidden_pair[1],
        "activation": activation_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_metrics": metrics_dict(prepared.y_train_raw, y_train_pred),
        "val_metrics": metrics_dict(prepared.y_val_raw, y_val_pred),
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
    }


def run_grid_search(
    prepared: PreparedData,
    epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    best_trial: dict[str, object] | None = None

    configs = list(itertools.product(
        DEFAULT_HIDDEN_PAIRS,
        DEFAULT_ACTIVATIONS,
        DEFAULT_BATCH_SIZES,
        DEFAULT_LEARNING_RATES,
    ))
    total_trials = len(configs)

    for i, (hidden_pair, activation_name, batch_size, learning_rate) in enumerate(configs, 1):
        print(
            f"  [{prepared.spec.slug}] Trial {i}/{total_trials}: "
            f"hidden={hidden_pair}, activation={activation_name}, batch={batch_size}, lr={learning_rate}...",
            end="\r",
            flush=True,
        )
        trial = train_one_config(
            prepared=prepared,
            hidden_pair=hidden_pair,
            activation_name=activation_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            device=device,
        )

        if best_trial is None or trial["val_metrics"]["rmse"] < best_trial["val_metrics"]["rmse"]:
            best_trial = trial
    
    print() # newline after the \r progress line

    if best_trial is None:
        raise RuntimeError(f"No trials completed for {prepared.spec.slug}")

    return best_trial


def build_metric_rows(prepared: PreparedData, best_trial: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for split_name, metrics_key in (("train", "train_metrics"), ("validation", "val_metrics")):
        metrics = best_trial[metrics_key]
        rows.append(
            {
                "dataset": prepared.spec.slug,
                "display_name": prepared.spec.display_name,
                "target": prepared.spec.target,
                "model": "PyTorch_4L",
                "split": split_name,
                "hidden1": best_trial["hidden1"],
                "hidden2": best_trial["hidden2"],
                "activation": best_trial["activation"],
                "batch_size": best_trial["batch_size"],
                "learning_rate": best_trial["learning_rate"],
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            }
        )
    return rows


def save_best_plots(prepared: PreparedData, best_trial: dict[str, object]) -> None:
    stem = f"{prepared.spec.slug}_{prepared.spec.target}_4l"
    train_plot = PLOTS_DIR / f"{stem}_train_pred_vs_actual.png"
    val_plot = PLOTS_DIR / f"{stem}_validation_pred_vs_actual.png"

    save_pred_vs_actual_plot(
        prepared.y_train_raw,
        best_trial["y_train_pred"],
        train_plot,
        f"{prepared.spec.display_name} | {prepared.spec.target} | 4L Train",
    )
    save_pred_vs_actual_plot(
        prepared.y_val_raw,
        best_trial["y_val_pred"],
        val_plot,
        f"{prepared.spec.display_name} | {prepared.spec.target} | 4L Validation",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch four-layer regression network on the Project 2 datasets.")
    parser.add_argument(
        "--dataset",
        choices=("all", "autompg", "house", "parkinsons_total", "parkinsons_motor"),
        default="all",
        help="Which dataset to run.",
    )
    parser.add_argument(
        "--include-motor",
        action="store_true",
        help="Include the extra Parkinsons motor_UPDRS target in the dataset pool.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs per trial.")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience per trial.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_outputs()
    set_seed()

    include_motor = args.include_motor or args.dataset == "parkinsons_motor"
    specs = load_dataset_specs(include_motor=include_motor)
    if args.dataset == "all":
        selected_specs = list(specs.values())
    else:
        if args.dataset not in specs:
            raise ValueError(f"Dataset '{args.dataset}' is not available with the current arguments.")
        selected_specs = [specs[args.dataset]]

    device = resolve_device(args.device)
    configure_runtime(device)
    print_device_status(args.device, device)

    metric_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []

    for spec in selected_specs:
        prepared = prepare_dataset(spec)
        best_trial = run_grid_search(prepared, epochs=args.epochs, patience=args.patience, device=device)

        metric_rows.extend(build_metric_rows(prepared, best_trial))
        config_rows.append(
            {
                "dataset": spec.slug,
                "display_name": spec.display_name,
                "target": spec.target,
                "hidden1": best_trial["hidden1"],
                "hidden2": best_trial["hidden2"],
                "activation": best_trial["activation"],
                "batch_size": best_trial["batch_size"],
                "learning_rate": best_trial["learning_rate"],
                "validation_r2": best_trial["val_metrics"]["r2"],
                "validation_rmse": best_trial["val_metrics"]["rmse"],
                "validation_mae": best_trial["val_metrics"]["mae"],
            }
        )
        save_best_plots(prepared, best_trial)

        print(
            f"[{spec.slug}] best hidden=({best_trial['hidden1']}, {best_trial['hidden2']}), "
            f"activation={best_trial['activation']}, batch={best_trial['batch_size']}, "
            f"lr={best_trial['learning_rate']}, val_rmse={best_trial['val_metrics']['rmse']:.4f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["dataset", "split"])
    configs_df = pd.DataFrame(config_rows).sort_values(["dataset"])

    metrics_path = OUTPUT_DIR / "metrics_4l.csv"
    configs_path = OUTPUT_DIR / "best_configs_4l.csv"
    metrics_df.to_csv(metrics_path, index=False)
    configs_df.to_csv(configs_path, index=False)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved configs to: {configs_path}")
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
