"""
Minimal entry-point for running the Docker-based SMPC/federated-learning
pipeline defined in `docker_smpc_manager.docker_deployer`.

Example
-------
python run_simulation.py data.csv --group-col client_id --target-col y --num-clients 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import subprocess
import sys
import itertools
import statsmodels.api as sm
import pandas as pd
import matplotlib

matplotlib.use("Agg")


def run_pipeline(df: pd.DataFrame, group_col: str = "client_id", target_col: str = "y", num_clients: int = 4):
    """
    Run the federated-learning SMPC simulation on the given DataFrame.
    Returns JSON-style data for frontend: df, correlation, r2, top relationships
    """
    from docker_smpc_manager import docker_deployer as dd
    from sklearn.linear_model import LinearRegression

    # ─── Build Docker Image ───
    dd.build_image()

    # ─── Keep only numeric columns ───
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available.")
    df = df[numeric_cols]

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # ─── Run Docker Simulation ───
    outputs = dd.fan_out_and_run(df, group_col, target_col, num_clients)
    local_models = dd.read_outputs(outputs)
    global_model = dd.average_models(local_models)
    df_pred = dd.apply_global_model(df.copy(), global_model, target_col)

    # ─── Convert prediction df to JSON for plotting ───
    df_json = df_pred.to_dict(orient="records")

    # ─── Correlation Matrix ───
    correlation = df.corr().fillna(0).to_dict()

    # ─── R² Calculation ───
    r2_results = []
    for target in df.columns:
        features = [c for c in df.columns if c != target]
        if not features:
            continue
        try:
            model = LinearRegression().fit(df[features], df[target])
            r2_results.append({
                "target": target,
                "features": features,
                "r2": round(model.score(df[features], df[target]), 4)
            })
        except Exception:
            continue

    # ─── Top 10 Relationships ───
    import statsmodels.api as sm
    import itertools
    results = []
    for target in df.columns:
        others = [c for c in df.columns if c != target]
        for x in others:
            corr_val = df[target].corr(df[x])
            results.append(f"corr | {target} ~ {x} = {corr_val:.3f}")
        for combo in itertools.combinations(others, 2):
            X = sm.add_constant(df[list(combo)])
            y = df[target]
            try:
                model = sm.OLS(y, X).fit()
                results.append(f"R2   | {target} ~ {', '.join(combo)} = {model.rsquared:.3f}")
            except:
                continue
    results = sorted(results, key=lambda r: abs(float(r.split('=')[-1])), reverse=True)
    top_10_results = results[:10]

    return df_json, correlation, r2_results, top_10_results

def _docker_available() -> bool:
    """Return True if the Docker daemon is reachable."""
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False

# High-level API
from docker_smpc_manager import docker_deployer as dd


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the federated-learning SMPC simulation end-to-end."
    )
    p.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=None,
        help="CSV containing merged data for all clients (optional; if omitted you will be prompted).",
    )
    p.add_argument(
        "--group-col",
        default="client_id",
        help="Column used to split the dataset into logical clients (default: client_id).",
    )
    p.add_argument(
        "--target-col",
        default="y",
        help="Name of the regression/target column (default: y).",
    )
    p.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of clients to simulate (default: 4).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_cli()

    # Ask interactively for the CSV path if it was not supplied on the CLI
    if args.csv is None:
        csv_input = input("Enter the path to the pooled CSV file: ").strip()
        args.csv = Path(csv_input)

    # Re‑prompt the user until a valid file is provided
    while not args.csv.exists():
        csv_input = input(f"File '{args.csv}' not found. Please re‑enter the path: ").strip()
        args.csv = Path(csv_input)

    # ── Verify that Docker is running
    if not _docker_available():
        print(
            "❌  Docker daemon is not accessible.\n"
            "    Please start Docker Desktop (or the Docker service) and rerun.\n"
        )
        sys.exit(1)

    # 1 ── Build (or rebuild) the Docker image that each compute node uses
    dd.build_image()

    # 2 ── Load the full dataset
    df = pd.read_csv(args.csv)

    # ── Limit to numeric columns (drop all string options)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print(" No numeric columns available for grouping or target.")
        sys.exit(1)

    print(f"Available numeric columns: {numeric_cols}")

    # ── Drop non-numeric columns and inform the user
    dropped_cols = set(df.columns) - set(numeric_cols)
    if dropped_cols:
        print(f"Dropping non-numeric columns: {sorted(dropped_cols)}")
        df.drop(columns=dropped_cols, inplace=True)

    # Prompt for group column among numeric-only
    while args.group_col not in numeric_cols:
        args.group_col = input(
            "Select a numeric column to split clients by: "
        ).strip()

    # Prompt for target column among numeric-only
    while args.target_col not in numeric_cols:
        args.target_col = input(
            "Select the numeric target/outcome column: "
        ).strip()

    # 3 ── Fan out, run each node, and collect JSON artefacts
    outputs = dd.fan_out_and_run(
        df=df,
        group_col=args.group_col,
        target_col=args.target_col,
        num_clients=args.num_clients,
    )

    # 4 ── Read local models and compute the federated (global) model
    local_models = dd.read_outputs(outputs)
    global_model = dd.average_models(local_models)

    # 5 ── Apply the global model to the entire dataset
    df_pred = dd.apply_global_model(df.copy(), global_model, args.target_col)

    pred_path = Path("predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"✔ Predictions written to {pred_path.resolve()}")
    print("✔ Figures stored in ./figures/")

    # ── Exploratory diagnostics: find strongest X→Y relationships
    def explore_relationships(df: pd.DataFrame, max_pred: int = 3):
        """
        For each numeric target Y and each combination of 1..max_pred numeric predictors X,
        compute Pearson correlation if len(X)==1, else OLS R² for Y~X1+…+Xk.
        Returns a sorted list of (target, predictors, metric, value).
        """
        numeric = df.select_dtypes(include="number")
        cols = numeric.columns.tolist()
        results = []
        for target in cols:
            others = [c for c in cols if c != target]
            # Single predictor: Pearson correlation
            for x in others:
                corr = numeric[target].corr(numeric[x])
                results.append((target, (x,), "corr", corr))
            # Multi-predictor: OLS R²
            for k in range(2, max_pred + 1):
                for combo in itertools.combinations(others, k):
                    X = numeric[list(combo)]
                    X = sm.add_constant(X)
                    y = numeric[target]
                    model = sm.OLS(y, X, missing="drop").fit()
                    results.append((target, combo, "R2", model.rsquared))
        # Sort by absolute metric value descending
        results.sort(key=lambda t: abs(t[3]), reverse=True)
        return results

    print("\nTop 10 relationships (corr or R²):")
    for target, preds, metric, value in explore_relationships(df)[:10]:
        preds_str = ", ".join(preds)
        print(f"  {metric:>4} | {target} ~ {preds_str} = {value:.3f}")

    # ── Additional Visualizations ────────────────────────────────────
    import matplotlib.pyplot as plt

    # 1. Correlation heatmap of all numeric features
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    fig.colorbar(cax, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig("static/figures/correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # 2. Boxplots for each numeric feature
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([df[col].dropna() for col in numeric_cols], labels=numeric_cols)
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_title("Boxplots of Numeric Features")
    fig.tight_layout()
    fig.savefig("static/figures/boxplots_all_features.png", dpi=300)
    plt.close(fig)

    # 3. Boxplot (candle-like) of target by group
    groups = sorted(df[args.group_col].unique())
    data_by_group = [df[df[args.group_col] == g][args.target_col] for g in groups]
    fig, ax = plt.subplots(figsize=(max(6, len(groups)), 4))
    ax.boxplot(data_by_group, labels=groups)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_title(f"Distribution of {args.target_col} by {args.group_col}")
    fig.tight_layout()
    fig.savefig(f"static/figures/{args.target_col}_by_{args.group_col}_boxplot.png", dpi=300)
    plt.close(fig)
    # ────────────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.linear_model import LinearRegression
import json

def simulate_data_analysis(df: pd.DataFrame):
    # 1. Convert DataFrame to JSON for boxplots
    df_json = df.to_dict(orient="records")

    # 2. Correlation matrix
    corr = df.corr(numeric_only=True)
    correlation_json = corr.fillna(0).to_dict()

    # 3. R² Scores
    r2_results = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    for target in numeric_cols:
        X_cols = [col for col in numeric_cols if col != target]
        if not X_cols:
            continue
        X = df[X_cols]
        y = df[target]
        model = LinearRegression()
        try:
            model.fit(X, y)
            score = model.score(X, y)
            r2_results.append({
                "target": target,
                "features": X_cols,
                "r2": round(score, 4)
            })
        except Exception:
            continue

    return df_json, correlation_json, r2_results


if __name__ == "__main__":
    main()
