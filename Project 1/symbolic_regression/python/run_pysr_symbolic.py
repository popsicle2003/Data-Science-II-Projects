# run_pysr_symbolic.py
# ============================================================
# Symbolic Regression (PySR) for:
#   - Auto MPG: mpg
#   - House Price: House_Price
#   - Parkinsons: motor_UPDRS and total_UPDRS
#
# Outputs:
#   outputs/
#     metrics_pysr.csv
#     equations_pysr.txt
#     plots/
#       <dataset>_<target>_pysr_pred_vs_actual.png
#     cleaned_for_scala/
#       autompg_xy.csv
#       house_xy.csv
#       parkinsons_total_xy.csv
#       parkinsons_motor_xy.csv
#     feature_maps/
#       feature_map_<dataset>_<target>_pysr.csv
# ============================================================

from __future__ import annotations

import os
import re
import json
import csv
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---- User file paths (your uploaded files) ----
AUTO_MPG_PATH = r"Raw Datasets\auto-mpg\auto-mpg.data-original"
HOUSE_PATH    = r"Raw Datasets\house_price_regression_dataset\house_price_regression_dataset.csv"
PARK_PATH     = r"Raw Datasets\parkinsons_telemonitoring\parkinsons_updrs.data"


# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Output dirs ----
OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CLEAN_DIR = os.path.join(OUT_DIR, "cleaned_for_scala")
FEATURE_MAP_DIR = os.path.join(OUT_DIR, "feature_maps")

import shutil

def clear_outputs() -> None:
    # Delete previous outputs entirely, then recreate folders
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(FEATURE_MAP_DIR, exist_ok=True)

# Call once at startup
clear_outputs()


os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(FEATURE_MAP_DIR, exist_ok=True)

# ---- PySR import (deferred-friendly error message) ----
try:
    from pysr import PySRRegressor
except Exception as e:
    PySRRegressor = None
    _pysr_import_error = e


@dataclass
class DatasetSpec:
    name: str
    target: str
    X: pd.DataFrame
    y: pd.Series


def parse_auto_mpg(path: str) -> pd.DataFrame:
    """
    Parses UCI auto-mpg.data-original format:
      mpg cylinders displacement horsepower weight acceleration model_year origin "car name"
    Handles missing as:
      mpg may appear as 'NA'
      horsepower may appear as '?'
    """
    rows = []
    # Capture 8 numeric-like tokens then a quoted string (car name)
    # Example line:
    # 18.0 8. 307.0 130.0 3504. 12.0 70. 1. "chevrolet chevelle malibu"
    pat = re.compile(
        r'^\s*([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+"(.*)"\s*$'
    )
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                # If a line doesn't match, skip with warning (should be rare)
                warnings.warn(f"Skipping unparsable line: {line[:80]}...")
                continue
            (mpg, cyl, disp, hp, wt, acc, year, origin, name) = m.groups()
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

    # Convert numeric columns; coerce errors to NaN
    for col in ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]:
        df[col] = pd.to_numeric(df[col].replace({"?": np.nan, "NA": np.nan}), errors="coerce")

    return df


def preprocess_auto_mpg(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Agreed rules:
      - Target: mpg
      - Drop rows with missing mpg
      - Median-impute horsepower
      - Drop car_name (ID-like string)
      - One-hot encode origin (categorical-ish)
    """
    # Drop missing target
    df = df.dropna(subset=["mpg"]).copy()

    # Median impute horsepower
    hp_median = df["horsepower"].median(skipna=True)
    df["horsepower"] = df["horsepower"].fillna(hp_median)

    # Drop car name
    df = df.drop(columns=["car_name"])

    # One-hot origin
    df["origin"] = df["origin"].astype(int)
    origin_oh = pd.get_dummies(df["origin"], prefix="origin", drop_first=False)
    df = pd.concat([df.drop(columns=["origin"]), origin_oh], axis=1)

    y = df["mpg"].astype(float)
    X = df.drop(columns=["mpg"]).astype(float)
    return X, y


def preprocess_house(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["House_Price"].astype(float)
    X = df.drop(columns=["House_Price"]).astype(float)
    return X, y


def preprocess_parkinsons(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[target].astype(float)

    # Drop the target from features; keep everything else numeric
    X = df.drop(columns=[target]).copy()

    # Ensure numeric; "subject#" is an integer identifier, but we keep it as numeric (it may help)
    X = X.apply(pd.to_numeric, errors="coerce")

    # No missing expected; drop rows if any appear due to coercion
    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep].astype(float)
    y = y.loc[keep].astype(float)
    return X, y


def standard_split_scale(
    X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    80/20 split; standardize features using training-only fit.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=seed, shuffle=True
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def save_pred_vs_actual_plot(y_true: np.ndarray, y_pred: np.ndarray, outpath: str, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    # Reference line
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def save_feature_map(dataset: str, target: str, feature_names: List[str]) -> Tuple[str, str]:
    """
    Saves mapping: x0, x1, ... -> original feature name.
    Returns (csv_path, pretty_text_block).
    """
    csv_path = os.path.join(FEATURE_MAP_DIR, f"feature_map_{dataset}_{target}_pysr.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "feature_name"])
        for i, name in enumerate(feature_names):
            w.writerow([f"x{i}", str(name)])

    lines = [
        "Feature map (PySR variable -> original feature):",
        f"[saved] {csv_path}",
        *[f"  x{i} -> {name}" for i, name in enumerate(feature_names)],
    ]
    return csv_path, "\n".join(lines)


def run_pysr_case(spec: DatasetSpec) -> Dict[str, object]:
    if PySRRegressor is None:
        raise RuntimeError(
            "PySR is not installed/available.\n"
            "Install with:\n"
            "  pip install pysr\n"
            "and ensure Julia is installed (PySR will guide/auto-install some parts).\n"
            f"Original import error: {_pysr_import_error}"
        )

    # IMPORTANT: keep feature order consistent with the matrix passed to PySR
    feature_names = list(spec.X.columns)

    X_train_s, X_test_s, y_train, y_test, _ = standard_split_scale(spec.X, spec.y)

    # Conservative operator set for stability + interpretability
    model = PySRRegressor(
        niterations=200,                  # increase (e.g., 500–2000) if you have time
        population_size=50,
        maxsize=20,                       # complexity cap
        maxdepth=8,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "log1p"],
        # Avoid invalid operations (e.g., sqrt of negative) via internal constraints
        constraints={
            "/": (-1, 9),                 # modest constraint
            "sqrt": 5,
            "log1p": 5,
        },
        loss="loss(x, y) = (x - y)^2",
        random_state=RANDOM_SEED,
        model_selection="best",            # pick best by internal score
        verbosity=1,
    )

    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    m = metrics_dict(y_test, y_pred)

    # Best equation as string
    eq = str(model.get_best())

    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{spec.name}_{spec.target}_pysr_pred_vs_actual.png")
    save_pred_vs_actual_plot(y_test, y_pred, plot_path, f"{spec.name} | {spec.target} | PySR")

    # NEW: save x-index -> feature mapping (for interpreting equations like x4, x18, ...)
    feature_map_path, feature_map_block = save_feature_map(spec.name, spec.target, feature_names)

    return {
        "dataset": spec.name,
        "target": spec.target,
        "language": "python",
        "model": "PySR",
        "metrics": m,
        "equation": eq,
        "plot_path": plot_path,
        "feature_map_path": feature_map_path,
        "feature_map_block": feature_map_block,
    }


def export_clean_xy_for_scala(name: str, X: pd.DataFrame, y: pd.Series, outname: str) -> str:
    """
    Exports numeric matrix with y as last column, header included.
    ScalaTion MatrixD.load can skip header row via '1' parameter.
    """
    xy = X.copy()
    xy["y"] = y.values
    outpath = os.path.join(CLEAN_DIR, outname)
    xy.to_csv(outpath, index=False)
    return outpath


def main() -> None:
    results = []

    # ---------- Auto MPG ----------
    auto_raw = parse_auto_mpg(AUTO_MPG_PATH)
    X_auto, y_auto = preprocess_auto_mpg(auto_raw)
    export_clean_xy_for_scala("autompg", X_auto, y_auto, "autompg_xy.csv")
    results.append(DatasetSpec(name="autompg", target="mpg", X=X_auto, y=y_auto))

    # ---------- House Price ----------
    X_house, y_house = preprocess_house(HOUSE_PATH)
    export_clean_xy_for_scala("house", X_house, y_house, "house_xy.csv")
    results.append(DatasetSpec(name="house", target="House_Price", X=X_house, y=y_house))

    # ---------- Parkinsons (2 targets) ----------
    X_pt, y_pt = preprocess_parkinsons(PARK_PATH, target="total_UPDRS")
    export_clean_xy_for_scala("parkinsons", X_pt, y_pt, "parkinsons_total_xy.csv")
    results.append(DatasetSpec(name="parkinsons", target="total_UPDRS", X=X_pt, y=y_pt))

    X_pm, y_pm = preprocess_parkinsons(PARK_PATH, target="motor_UPDRS")
    export_clean_xy_for_scala("parkinsons", X_pm, y_pm, "parkinsons_motor_xy.csv")
    results.append(DatasetSpec(name="parkinsons", target="motor_UPDRS", X=X_pm, y=y_pm))

    # ---------- Run PySR on all cases ----------
    metrics_rows = []
    equations_lines = []

    for spec in results:
        out = run_pysr_case(spec)
        metrics_rows.append(
            {
                "dataset": out["dataset"],
                "target": out["target"],
                "language": out["language"],
                "model": out["model"],
                "r2": out["metrics"]["r2"],
                "rmse": out["metrics"]["rmse"],
                "mae": out["metrics"]["mae"],
                "plot_path": out["plot_path"],
                "feature_map_path": out["feature_map_path"],  # NEW
            }
        )

        # NEW: include feature-map block in the equations text file for easy reporting
        equations_lines.append(
            f"{out['dataset']} | {out['target']} | PySR best equation:\n"
            f"{out['equation']}\n\n"
            f"{out['feature_map_block']}\n"
        )

    # Save summary outputs
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["dataset", "target"])
    metrics_path = os.path.join(OUT_DIR, "metrics_pysr.csv")
    metrics_df.to_csv(metrics_path, index=False)

    eq_path = os.path.join(OUT_DIR, "equations_pysr.txt")
    with open(eq_path, "w", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("PySR equations + feature maps\n")
        f.write("=" * 100 + "\n\n")
        f.write("\n" + ("-" * 100 + "\n\n").join(equations_lines))

    print("DONE ✅")
    print(f"- Metrics: {metrics_path}")
    print(f"- Equations: {eq_path}")
    print(f"- Plots: {PLOTS_DIR}")
    print(f"- Feature maps: {FEATURE_MAP_DIR}")
    print(f"- Cleaned CSVs for ScalaTion: {CLEAN_DIR}")


if __name__ == "__main__":
    main()
