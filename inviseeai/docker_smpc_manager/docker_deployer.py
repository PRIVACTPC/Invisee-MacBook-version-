from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import numpy as np
import pandas as pd
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_NAME = "inviseeai-client1:latest"


# ----------  Visualisation helpers (Matplotlib‑only) ----------
from pathlib import Path as _Path  # alias to avoid shadowing above import


def _figdir() -> _Path:
    """Ensure a ./figures directory exists and return its Path."""
    d = _Path.cwd() / "figures"
    d.mkdir(exist_ok=True)
    return d


def _heatmap_local_coeffs(json_paths):
    """Plot a heat‑map of each client’s coefficients."""
    if not json_paths:
        return
    import json as _json, pandas as _pd

    frames, ids = [], []
    for p in json_paths:
        with open(p) as f:
            payload = _json.load(f)
        frames.append(_pd.Series(payload["coefficients"]))
        ids.append(p.stem)

    mat = _pd.concat(frames, axis=1)
    mat.columns = ids

    plt.figure(figsize=(8, 4))
    plt.imshow(mat.values, aspect="auto")
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.title("Local model coefficient heat‑map")
    plt.colorbar(label="Coefficient value")
    plt.tight_layout()
    plt.savefig(_figdir() / "local_coefficients_heatmap.png", dpi=300)
    plt.close()


def _bar_global_coeffs(averaged_coeffs):
    """Plot a bar chart of the federated (global) coefficients."""
    import pandas as _pd

    coef = _pd.Series(averaged_coeffs)
    plt.figure(figsize=(8, 3))
    plt.bar(coef.index, coef.values)
    plt.ylabel("Weight")
    plt.title("Global model coefficients")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(_figdir() / "global_coefficients_bar.png", dpi=300)
    plt.close()
# ----------------------------------------------------------------


def _sh(*cmd: str) -> None:
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Docker command failed:", e)
        raise


def build_image() -> None:
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    build_context = Path(__file__).parent
    print("(re)building Docker image...")
    _sh(
        "docker", "build",
        "-f", str(dockerfile_path.resolve()),
        "-t", IMAGE_NAME,
        str(build_context.resolve())
    )


def _run_container(input_path: Path, output_path: Path, idx: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)
    cname = f"client_{idx}_{uuid.uuid4().hex[:6]}"
    _sh(
        "docker", "run", "--rm",
        "-v", f"{input_path.resolve().as_posix()}:/data/input.csv:ro",
        "-v", f"{output_path.resolve().as_posix()}:/data/output.json:rw",
        "--name", cname,
        IMAGE_NAME,
        "python", "compute_node.py",
        "/data/input.csv",
        "/data/output.json",
    )



def fan_out_and_run(
        df: pd.DataFrame,
        group_col: str,
        target_col: str,
        num_clients: int,
) -> List[Path]:
    outputs: List[Path] = []
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        groups = df[group_col].unique()[: num_clients]

        for idx, g in enumerate(groups, 1):
            frag = df[df[group_col] == g].copy()
            from .data_handler import sanitize
            frag = sanitize(frag)

            if target_col not in frag.columns:
                print(f"group {g}: target column missing, skip")
                continue

            cols = [c for c in frag.columns if c != target_col] + [target_col]
            in_csv = tmp / f"input_{idx}.csv"
            frag[cols].to_csv(in_csv, index=False)

            out_json = tmp / f"output_{idx}.json"
            outputs.append(out_json)

            _run_container(in_csv, out_json, idx)

        safe_outputs: List[Path] = []
        for src in outputs:
            if src.exists():
                dest = Path.cwd() / src.name
                shutil.copy(src, dest)
                safe_outputs.append(dest)
            else:
                print(f"missing output {src}")

        _heatmap_local_coeffs(safe_outputs)  # visualise local models
        return safe_outputs


def read_outputs(paths: List[Path]) -> List[Dict]:
    models = []
    for p in paths:
        try:
            with open(p) as f:
                models.append(json.load(f))
        except FileNotFoundError:
            print(f"missing output {p}")
    return models


def average_models(models: List[Dict]) -> Dict:
    if not models:
        raise ValueError("no local models")

    coeffs = {}
    for m in models:
        for k, v in m["coefficients"].items():
            coeffs.setdefault(k, []).append(v)

    averaged = {
        k: float(np.mean(vs)) for k, vs in coeffs.items()
    }
    intercept = float(np.mean([m["intercept"] for m in models]))

    _bar_global_coeffs(averaged)            # visualise global model
    return {"intercept": intercept, "coefficients": averaged}


def apply_global_model(df: pd.DataFrame, model: Dict, target_col: str) -> pd.DataFrame:
    coef = model["coefficients"]
    present = [c for c in coef if c in df.columns]
    if not present:
        print("no matching features in dataset")
        return df

    y_pred = model["intercept"] + df[present].dot(pd.Series(coef)[present])
    df[target_col] = y_pred
    return df
