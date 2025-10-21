import os
from uuid import uuid4
from io import StringIO
from pathlib import Path
import traceback

import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for

from api_client_logic import get_client
from run_simulation import run_pipeline
from services import apply_k_anonymity, apply_l_contextual_mask, generate_summary, apply_date_shifting
from differential_privacy import mask_string_columns, apply_differential_privacy
from smpc_simulation import simulate_secure_sum, DockerSMPCError


app = Flask(__name__)
app.secret_key = 'secret_key_for_session'  # Replace with env var in production


class DatasetNotLoaded(RuntimeError):
    """Raised when a route requires a dataset but none is present in the session."""


_DATA_CACHE: dict[str, str] = {}


def _assign_token(name: str, data: str) -> str:
    """Store *data* under a fresh token, replacing any existing entry for *name*."""
    old = session.pop(name, None)
    if old:
        _DATA_CACHE.pop(old, None)
    token = uuid4().hex
    _DATA_CACHE[token] = data
    session[name] = token
    return token


def _update_token(name: str, data: str) -> str:
    """Update the cached value for *name*, initialising it if needed."""
    token = session.get(name)
    if not token or token not in _DATA_CACHE:
        return _assign_token(name, data)
    _DATA_CACHE[token] = data
    return token


def _get_token_data(name: str, *, required: bool = True) -> str | None:
    """Return cached data for *name*."""
    token = session.get(name)
    if not token:
        if required:
            raise DatasetNotLoaded(f"Missing session token '{name}'.")
        return None
    data = _DATA_CACHE.get(token)
    if data is None and required:
        raise DatasetNotLoaded(f"Session token '{name}' expired.")
    return data


def _current_csv_text(required: bool = True) -> str | None:
    legacy = session.pop("csv_data", None)
    if legacy is not None:
        legacy_original = session.pop("original_csv", None)
        _initialise_dataset(legacy)
        if legacy_original is not None:
            _assign_token("original_token", legacy_original)
    return _get_token_data("csv_token", required=required)


def _original_csv_text(required: bool = True) -> str | None:
    legacy = session.pop("original_csv", None)
    if legacy is not None:
        _assign_token("original_token", legacy)
    return _get_token_data("original_token", required=required)


def _load_working_df() -> pd.DataFrame:
    csv_text = _current_csv_text(required=True)
    if not csv_text:
        raise DatasetNotLoaded
    return pd.read_csv(StringIO(csv_text))


def _replace_working_csv(csv_text: str) -> None:
    _update_token("csv_token", csv_text)


def _initialise_dataset(csv_text: str) -> None:
    _assign_token("csv_token", csv_text)
    _assign_token("original_token", csv_text)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename.endswith(".csv"):
            data = file.read().decode("utf-8")
            df = pd.read_csv(StringIO(data))
            _initialise_dataset(data)
            session["filename"] = file.filename
            columns = df.columns.tolist()
            return render_template("select_columns.html", columns=columns)
    return render_template("index.html")

@app.route("/preview", methods=["GET", "POST"])
def preview():
    if request.method == "POST":
        selected_columns = request.form.getlist("columns")
        if not selected_columns:
            return "No columns selected", 400

        try:
            df = _load_working_df()
        except DatasetNotLoaded:
            flash("No dataset available. Please upload a file first.", "danger")
            return redirect(url_for("index"))

        filtered_df = df[selected_columns]
        _replace_working_csv(filtered_df.to_csv(index=False))
        table_html = filtered_df.head(100).to_html(classes="table table-striped", index=False)
        return render_template("preview.html", table=table_html)

    # If it's a GET request, just show current session data
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    table_html = df.head(100).to_html(classes="table table-striped", index=False)
    return render_template("preview.html", table=table_html)



@app.route("/k_form", methods=["GET"])
def k_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    columns = df.columns.tolist()
    return render_template("k_form.html", columns=columns)

@app.route("/k_anonymize", methods=["POST"])
def k_anonymize():
    selected_columns = request.form.getlist("quasi_columns")
    k = int(request.form["k_value"])
    try:
        csv_text = _current_csv_text()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("k_form"))

    try:
        result_df = apply_k_anonymity(csv_text, selected_columns, k)
        _replace_working_csv(result_df.to_csv(index=False))
        flash(f"k-Anonymity applied successfully with k = {k}.", "success")
    except Exception as e:
        flash(f"k-Anonymity failed: {e}", "danger")
        return redirect(url_for("k_form"))

    table_html = result_df.head(100).to_html(classes="table table-striped", index=False)
    return render_template("preview.html", table=table_html)


@app.route("/l_form", methods=["GET"])
def l_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    columns = df.columns.tolist()
    return render_template("l_form.html", columns=columns)

@app.route("/l_mask", methods=["POST"])
def l_mask():
    selected_columns = request.form.getlist("columns")
    l_value = int(request.form["l_value"])
    try:
        csv_text = _current_csv_text()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("l_form"))

    try:
        new_df = apply_l_contextual_mask(csv_text, selected_columns, l_value)
        _replace_working_csv(new_df.to_csv(index=False))
        flash(f"ℓ-Diversity masking applied successfully with ℓ = {l_value}.", "success")
    except Exception as e:
        flash(f"ℓ-Diversity masking failed: {e}", "danger")
        return redirect(url_for("l_form"))

    table_html = new_df.head(100).to_html(classes="table table-striped", index=False)
    return render_template("preview.html", table=table_html)

@app.route("/summary")
def summary():
    from summary_stats import simple_summary

    try:
        df = _load_working_df()

        # Generate summary statistics
        summary_data = simple_summary(df)

        return render_template("summary.html", summary=summary_data)

    except Exception as e:
        return f"Summary failed: {e}", 500



@app.route("/visualize")
def visualize():
    after_text = _current_csv_text(required=False)
    before_text = _original_csv_text(required=False)
    if not after_text or not before_text:
        flash("No dataset loaded. Please upload a file to begin.", "warning")
        return redirect(url_for('index'))

    try:
        df_after = pd.read_csv(StringIO(after_text))
        df_before = pd.read_csv(StringIO(before_text))

        # Use columns from the transformed version (assumes structure is preserved)
        columns = df_after.columns.tolist()

        # Convert both to JSON (used by Plotly in JS)
        df_json = df_after.to_json(orient='records')
        df_original_json = df_before.to_json(orient='records')

        # Pass both datasets to the template
        return render_template(
            "visualize.html",
            columns=columns,
            df_json=df_json,
            df_original_json=df_original_json
        )

    except Exception as e:
        flash(f"An error occurred while preparing the visualization: {e}", "danger")
        return redirect(url_for("preview"))








#
# @app.route("/visualize")
# def visualize():
#
#     # First, check if the data exists in the session to prevent errors.
#     if "csv_data" not in session:
#         flash("No dataset loaded. Please upload a file to begin.", "warning")
#         return redirect(url_for('index')) # Redirect to your homepage
#
#     try:
#         # Load the dataframe from the session CSV data
#         df = pd.read_csv(StringIO(session["csv_data"]))
#         columns = df.columns.tolist()
#
#         # --- THIS IS THE CRITICAL STEP ---
#         # Convert the entire DataFrame into a JSON string with a 'records' orientation.
#         # This creates a list of dictionaries, which is the perfect format for JavaScript.
#         df_json = df.to_json(orient='records')
#
#         # Pass both the list of columns and the full JSON data to the template.
#         # The variable name 'df_json' here MUST match the one in the HTML's <script> tag.
#         return render_template("visualize.html", columns=columns, df_json=df_json)
#
#     except Exception as e:
#         flash(f"An error occurred while preparing the visualization: {e}", "danger")
#         return redirect(url_for('preview')) # Or redirect to another safe page


# @app.route("/download")
# def download():
#     csv_data = session.get("csv_data")
#     if not csv_data:
#         return "No data available for download.", 400
#
#     return Response(
#         csv_data,
#         mimetype="text/csv",
#         headers={"Content-Disposition": "attachment;filename=anonymized_data.csv"}
#     )
# Set your secure passcode here
CORRECT_PASSCODE = "secure123"


@app.route("/download", methods=["GET", "POST"])
def download():
    csv_text = _current_csv_text(required=False)
    if not csv_text:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))

    df = pd.read_csv(StringIO(csv_text))

    # Get passcode from form or query param
    user_passcode = request.form.get("passcode") or request.args.get("passcode")

    if user_passcode == CORRECT_PASSCODE:
        # Return the current working dataset (already edited via the pipeline)
        result_csv = df.to_csv(index=False)
        filename = "edited_dataset.csv"
    else:
        # Apply extra masking and differential privacy for unauthenticated downloads
        masked_df = mask_string_columns(df)
        column_epsilons = {
            col: 1.0 for col in masked_df.select_dtypes(include="number").columns
        }
        dp_df = apply_differential_privacy(masked_df, column_epsilons)
        result_csv = dp_df.to_csv(index=False)
        filename = "anonymized_dataset.csv"

    # Return file for download
    return Response(
        result_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/t_form", methods=["GET"])
def t_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    columns = df.columns.tolist()
    return render_template("t_form.html", columns=columns)
@app.route("/t_closeness", methods=["POST"])
def t_closeness():
    quasi_columns = request.form.getlist("quasi_columns")
    sensitive_column = request.form["sensitive_column"]
    t_value = float(request.form["t_value"])

    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("t_form"))

    if not quasi_columns:
        flash("Please select at least one quasi-identifier.", "warning")
        return redirect(url_for("t_form"))

    try:
        from services import apply_t_closeness
        df = apply_t_closeness(df, quasi_columns, sensitive_column, t_value)
        _replace_working_csv(df.to_csv(index=False))
        flash(f"t-Closeness applied successfully with t = {t_value:.2f}.", "success")
    except Exception as e:
        flash(f"t-Closeness failed: {e}", "danger")
        return redirect(url_for("t_form"))

    table_html = df.head(100).to_html(classes="table table-striped", index=False)
    return render_template("preview.html", table=table_html)



@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/smpc_demo", methods=["GET"])
def smpc_demo_form():
    csv_text = _current_csv_text(required=False)
    if not csv_text:
        flash("Upload a dataset to access the SMPC secure-sum simulation.", "warning")
        return redirect(url_for("preview"))

    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        flash(f"Could not read the dataset: {exc}", "danger")
        return redirect(url_for("preview"))

    numeric_df = df.select_dtypes(include="number")
    numeric_columns = numeric_df.columns.tolist()
    if not numeric_columns:
        flash("The current dataset has no numeric columns available for SMPC.", "warning")
        return redirect(url_for("preview"))

    sample_preview = numeric_df.apply(pd.to_numeric, errors="coerce").to_dict(orient="records")

    return render_template(
        "smpc_demo.html",
        numeric_columns=numeric_columns,
        sample_table=sample_preview,
        total_rows=numeric_df.shape[0],
    )


@app.route("/run_smpc_demo", methods=["POST"])
def run_smpc_demo():
    csv_text = _current_csv_text(required=False)
    if not csv_text:
        flash("Upload a dataset to access the SMPC secure-sum simulation.", "warning")
        return redirect(url_for("preview"))

    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        flash(f"Could not read the dataset: {exc}", "danger")
        return redirect(url_for("preview"))

    numeric_df = df.select_dtypes(include="number")
    numeric_columns = numeric_df.columns.tolist()
    if not numeric_columns:
        flash("The current dataset has no numeric columns available for SMPC.", "warning")
        return redirect(url_for("preview"))

    column = request.form.get("dataset_column")
    if not column:
        flash("Select a numeric column to draw values from.", "warning")
        return redirect(url_for("smpc_demo_form"))

    if column not in numeric_df.columns:
        flash("Selected column not found or not numeric.", "danger")
        return redirect(url_for("smpc_demo_form"))

    series = pd.to_numeric(numeric_df[column], errors="coerce").dropna()
    values = series.tolist()
    if len(values) < 2:
        flash("The selected column does not contain valid numeric data.", "warning")
        return redirect(url_for("smpc_demo_form"))

    try:
        noise_range = float(request.form.get("noise_range") or 50.0)
    except ValueError:
        flash("Noise range must be a numeric value.", "danger")
        return redirect(url_for("smpc_demo_form"))

    seed_input = request.form.get("seed")
    seed = None
    if seed_input:
        try:
            seed = int(seed_input)
        except ValueError:
            flash("Seed must be an integer.", "danger")
            return redirect(url_for("smpc_demo_form"))

    rebuild_image = request.form.get("rebuild_image") == "on"

    try:
        result = simulate_secure_sum(
            values,
            seed=seed,
            rebuild_image=rebuild_image,
            noise_range=noise_range,
        )
    except (ValueError, DockerSMPCError) as exc:
        flash(str(exc), "danger")
        return redirect(url_for("smpc_demo_form"))
    except Exception:
        traceback.print_exc()
        flash("SMPC simulation failed due to an unexpected error.", "danger")
        return redirect(url_for("smpc_demo_form"))

    aggregator_view = {
        "masked_contributions": [p.masked_contribution for p in result.parties],
        "masked_sum": result.masked_sum,
        "recovered_sum": result.recovered_sum,
    }

    return render_template(
        "smpc_result.html",
        result=result,
        aggregator_view=aggregator_view,
        numeric_columns=numeric_columns,
        chosen_values=values,
        selected_column=column,
    )


@app.route("/federated_form", methods=["GET"])
def federated_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    columns = df.columns.tolist()
    return render_template("federated_form.html", columns=columns)

#
# @app.route("/federated_mask", methods=["POST"])
# def federated_mask():
#     from federated_data_masker import federated_feature_masking
#
#     group_col = request.form.get("group_column")
#     selected_cols = request.form.getlist("mask_columns")
#     mask_mode = request.form.get("mask_mode")
#
#     df = pd.read_csv(StringIO(session["csv_data"]))
#
#     try:
#         masked_df = federated_feature_masking(df, group_col, selected_cols, mask_mode)
#         session["csv_data"] = masked_df.to_csv(index=False)
#         table_html = masked_df.head(100).to_html(classes="table table-striped", index=False)
#         return render_template("preview.html", table=table_html)
#     except Exception as e:
#         return "Federated masking failed: {e}", 500

@app.route("/federated_pipeline_form", methods=["GET"])
def federated_pipeline_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    columns = df.columns.tolist()
    return render_template("federated_pipeline_form.html", columns=columns)
@app.route("/run_federated_pipeline", methods=["POST"])
def run_federated_pipeline():
    from docker_smpc_manager.docker_deployer import (
        build_image, fan_out_and_run, read_outputs, average_models, apply_global_model
    )

    mode = request.form["mode"]
    num_clients = int(request.form["num_clients"])
    group_col = request.form["group_column"]
    target_col = request.form["target_column"]

    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))

    try:
        # Process mode
        if mode == "numeric":
            df = df.select_dtypes(include=["number"])
        elif mode == "date":
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df.drop(columns=[col], inplace=True)
                except Exception:
                    continue
        else:
            return "Invalid mode selected.", 400

        # Validate required columns
        if group_col not in df.columns:
            return f"Group column '{group_col}' not found in dataset.", 400
        if target_col not in df.columns:
            return f"Target column '{target_col}' not found in dataset.", 400

        print("[INFO] Building Docker image...")
        build_image()

        print("[INFO] Running federated containers...")
        fragments = fan_out_and_run(df, group_col, target_col, num_clients)

        print("[INFO] Reading outputs...")
        outputs = read_outputs(fragments)

        if not outputs:
            return "No output from containers.", 500

        print("[INFO] Averaging models...")
        global_model = average_models(outputs)

        print("[INFO] Applying global model...")
        adjusted_df = apply_global_model(df, global_model, target_col)

        _replace_working_csv(adjusted_df.to_csv(index=False))
        Path("client_outputs").mkdir(parents=True, exist_ok=True)
        adjusted_df.to_csv("client_outputs/federated_adjusted.csv", index=False)

        table_html = adjusted_df[[target_col]].head(100).to_html(classes="table table-striped", index=False)

        return render_template("federated_results.html", table=table_html, model=global_model)

    except Exception:
        traceback.print_exc()
        return f"Federated pipeline failed:<pre>{traceback.format_exc()}</pre>", 500


@app.route("/knmi", methods=["GET", "POST"])
def knmi_flow():
    client = get_client("KNMI")
    client.api_key = os.getenv("KNMI_API_KEY")

    if request.method == "GET" and not request.args.get("resume"):
        # Reset all KNMI-related state
        session.pop("knmi_dataset", None)
        session.pop("knmi_version", None)
        session.pop("knmi_filename", None)

    step = "dataset"
    dataset = session.get("knmi_dataset")
    version = session.get("knmi_version")
    filename = session.get("knmi_filename")
    files = []
    versions = []
    if request.method == "POST":
        if "dataset" in request.form:
            session["knmi_dataset"] = request.form["dataset"]
            dataset = session["knmi_dataset"]
            step = "version"

        elif "version" in request.form:
            session["knmi_version"] = request.form["version"]
            version = session["knmi_version"]
            step = "file"

        elif "filename" in request.form:
            session["knmi_filename"] = request.form["filename"]
            filename = session["knmi_filename"]
            step = "confirm"

        elif "confirm" in request.form:
            path = client.download_file(dataset, filename, version)
            with open(path, "r", encoding="utf-8") as f:
                csv_text = f.read()
            _initialise_dataset(csv_text)
            if filename:
                session["filename"] = filename
            flash(f" File '{filename}' loaded into session.", "success")
            return redirect(url_for("preview"))

    if dataset and not version:
        step = "version"
        versions = {
            "Actuele10mindataKNMIstations": ["2"],
            "radar_rainfall": ["1"],
            "synopdata": ["1"],
            "daggegevens": ["1"]
        }.get(dataset, [])

    elif dataset and version and not filename:
        step = "file"
        files = client.list_files(dataset, version)

    elif dataset and version and filename:
        step = "confirm"

    return render_template(
        "knmi_form.html",
        step=step,
        dataset=dataset,
        version=version,
        filename=filename,
        datasets=[
            "Actuele10mindataKNMIstations",
            "radar_rainfall",
            "synopdata",
            "daggegevens"
        ],
        versions=versions,
        files=files
    )


@app.route("/noise_form", methods=["GET", "POST"])
def noise_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))
    columns = df.columns.tolist()
    return render_template("noise_form.html", columns=columns)
#
# @app.route("/inject_noise", methods=["POST"])
# def inject_noise():
#     import pandas as pd
#     from io import StringIO
#     from noise_injector import apply_noise
#
#     csv_text = session.get("csv_data")
#     if not csv_text:
#         flash("No data available. Please upload or load a dataset first.", "warning")
#         return redirect(url_for("index"))
#
#     df = pd.read_csv(StringIO(csv_text))
#
#     # Build configs from form data
#     configs = []
#     for col in request.form.getlist("column"):
#         configs.append({
#             "column": col,
#             "noise": request.form.get(f"noise_type_{col}"),
#             "percentage": request.form.get(f"percentage_{col}", 0),
#             "value": request.form.get(f"custom_value_{col}") or None
#         })
#
#     try:
#         df_noised, log = apply_noise(df, configs)
#         session["csv_data"] = df_noised.to_csv(index=False)
#         flash("Noise injection applied: " + "; ".join(log), "success")
#         table_html = df_noised.head(100).to_html(classes="table table-striped", index=False)
#         return render_template("preview.html", table=table_html)
#     except Exception as e:
#         flash(f"Error during noise injection: {str(e)}", "danger")
#         return redirect(url_for("noise_form"))

@app.route("/docker_sim_form", methods=["GET"])
def docker_sim_form():
    csv_text = _current_csv_text(required=False)
    if not csv_text:
        flash("Please upload data first.", "warning")
        return redirect(url_for("index"))

    df = pd.read_csv(StringIO(csv_text))
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return render_template("docker_sim_form.html", columns=columns, numeric_cols=numeric_cols)

@app.route("/run_docker_simulation", methods=["POST"])
def run_docker_simulation():
    csv_text = _current_csv_text(required=False)
    if not csv_text:
        flash("No dataset found in session", "danger")
        return redirect(url_for("index"))

    group_col = request.form.get("group_col")
    target_col = request.form.get("target_col")
    df = pd.read_csv(StringIO(csv_text))
    df = df.select_dtypes(include="number")

    try:
        df_json, correlation_json, r2_data, top_relationships = run_pipeline(
             df=df,
              group_col=group_col,
             target_col=target_col
         )
    except Exception as e:
        return f"Simulation failed: {e}", 500

    return render_template(
        "docker_sim_result.html",
        df_json=df_json,
        correlation_json=correlation_json,
        r2_data=r2_data,
        top_relationships=top_relationships,
        group_col=group_col,
        target_col=target_col
    )
import warnings # Add this import at the top of your app.py

@app.route("/date_shift_form", methods=["GET"])
def date_shift_form():
    try:
        df = _load_working_df()
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))

    all_columns = df.columns.tolist()
    date_columns = []
    sample_size = min(100, len(df))
    
    for col in all_columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
            
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().sample(n=min(len(df[col].dropna()), 5))
                if not sample.empty:
                    # --- NEW: Temporarily suppress the warning here ---
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        pd.to_datetime(sample, errors='raise')
                    # --------------------------------------------------
                    date_columns.append(col)
            except (ValueError, TypeError):
                continue

    return render_template(
        "date_shift_form.html",
        columns=all_columns,
        date_columns=date_columns
    )

@app.route("/apply_date_shift", methods=["POST"])
def apply_date_shift():
    try:
        id_col = request.form["id_column"]
        date_cols = request.form.getlist("date_columns")
        max_days = int(request.form["max_shift_days"])
        csv_text = _current_csv_text()
    except (KeyError, ValueError):
        flash("Invalid form submission. Please fill out all required fields correctly.", "danger")
        return redirect(url_for("date_shift_form"))
    except DatasetNotLoaded:
        flash("No dataset available. Please upload a file first.", "danger")
        return redirect(url_for("index"))

    if not date_cols:
        flash("You must select at least one date column to shift.", "warning")
        return redirect(url_for("date_shift_form"))

    try:
        result_df = apply_date_shifting(csv_text, id_col, date_cols, max_days)
        _replace_working_csv(result_df.to_csv(index=False))
        flash(f"Date shifting applied successfully with a max shift of {max_days} days.", "success")
    except Exception as e:
        flash(f"Date shifting failed: {e}", "danger")
        return redirect(url_for("date_shift_form"))

    table_html = result_df.head(100).to_html(classes="table table-striped", index=False)
    return render_template("preview.html", table=table_html)




if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=500)
