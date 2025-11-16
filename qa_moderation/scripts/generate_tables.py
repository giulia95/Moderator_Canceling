"""Module for generating LaTeX tables from metrics data."""

import json

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pandas import DataFrame, Series


# Configuration
OUTPUT_DIR = Path("output")
INCLUDE_ALL = False  # Control whether to include "All" metrics

# LaTeX formatting constants
LATEX_PREAMBLE = """\
\\documentclass[border=10pt]{standalone}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{courier}"""

LATEX_BEGIN = """\
\\begin{document}
\\renewcommand{\\arraystretch}{1.3}"""

# Model definitions in desired display order
MODELS: Dict[str, str] = {
    # Baselines (in order)
    "PKU-Alignment/beaver-dam-7b": "beaver-dam-7b",
    "meta-llama/Llama-Guard-4-12B": "Llama Guard 4 12B (ICL)",
    "meta-llama/Llama-Guard-3-8B": "Llama Guard 3 8B (ICL)",
    # Fine-tuned models (in order)
    "saiteki-kai/QA-Llama-Guard-3-8B": "Llama Guard 3 8B (FT)",
    "saiteki-kai/QA-Llama-3.1": "Llama 3.1 8B (FT)",
    "saiteki-kai/QA-DeBERTa-v3-large": "DeBERTa v3 large (FT)",
    "saiteki-kai/QA-DeBERTa-v3-large-threshold": "DeBERTa Threshold (FT)",
}

# Metrics configuration
METRICS_CONFIG: Dict[str, Dict[str, Any]] = {
    "flagged": {
        "metrics": ["flagged/f1", "flagged/aucpr", "flagged/fpr"],
        "display": [("F1", True), ("AUPRC", True), ("FPR", False)],
    },
    "average": {
        "metrics": ["micro_f1", "macro_f1"],
        "display": [("Micro-F1", True), ("Macro-F1", True)],
    },
}

LANGUAGES = ["all", "english", "italian"]
LANG_DISPLAY = {"english": "ENG", "italian": "ITA", "all": "All"}


def load_metrics_data(models: Dict[str, str], languages: List[str], outdir: Path) -> Dict[str, DataFrame]:
    """Load metrics data for all models and languages."""
    dfs: Dict[str, DataFrame] = {}
    default_metrics = {
        "flagged/f1": 0.0,
        "flagged/aucpr": 0.0,
        "flagged/fpr": 0.0,
        "micro_f1": 0.0,
        "macro_f1": 0.0,
    }

    for lang in languages:
        df_metrics = pd.DataFrame()
        for model_name, name in models.items():
            model_name = model_name.replace("/", "__")
            filepath = Path(outdir / f"metrics/{model_name}_{lang}.json")
            try:
                if filepath.exists():
                    metrics = json.load(filepath.open())
                    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
                else:
                    print(f"Missing {filepath}, using default values")
                    metrics_df = pd.DataFrame([default_metrics.copy()])
                metrics_df["model"] = name
                df_metrics = pd.concat([df_metrics, metrics_df])
            except Exception as e:
                print(f"Error processing {name}: {e}")
                metrics_df = pd.DataFrame([default_metrics.copy()])
                metrics_df["model"] = name
                df_metrics = pd.concat([df_metrics, metrics_df])
        dfs[lang] = df_metrics
    return dfs


def create_metrics_table(dfs: Dict[str, DataFrame], metrics: List[str], include_all: bool = True) -> DataFrame:
    """Create a table with the specified metrics and languages."""
    langs = LANGUAGES if include_all else [lang for lang in LANGUAGES if lang != "all"]
    model_names = list(MODELS.values())
    table = pd.DataFrame(index=pd.Index(model_names, name="model"))

    for metric in metrics:
        for lang in langs:
            col_name = f"{metric.split('/')[-1]}_{lang}"
            series = dfs[lang].set_index("model")[metric]
            table[col_name] = series.reindex(model_names).fillna(0.0)

    return table


def format_value_with_bold(value: float, is_best: bool) -> str:
    """Format a numeric value, making it bold if it's the best in its column."""
    return f"\\textbf{{{value:.3f}}}" if is_best else f"{value:.3f}"


def find_best_values(df: DataFrame, maximize: bool = True) -> Series:
    """Find the best value in each column."""
    return df.max() if maximize else df.min()


def export_table_to_latex_standalone(df: DataFrame, filename: Path, metric_config: Dict[str, Any]) -> None:
    """Export table to standalone LaTeX document with proper formatting."""
    # Ensure correct model order
    model_names = list(MODELS.values())
    formatted_df = df.reindex(model_names)

    # Process each column separately to handle best values
    for col in formatted_df.columns:
        # Determine if this is FPR metric (where lower is better)
        metric_type = col.split("_")[0]  # e.g., "fpr", "f1", "aucpr"
        is_fpr = metric_type == "fpr"

        # Find best value for this column
        values = formatted_df[col]
        best_value = values.min() if is_fpr else values.max()

        # Bold all values that match the best
        formatted_df[col] = [format_value_with_bold(val, abs(val - best_value) < 1e-3) for val in values]

    # Generate LaTeX
    table_code = formatted_df.to_latex(
        index=True,
        multirow=False,
        escape=False,
        bold_rows=False,
        column_format="l" + "c" * len(formatted_df.columns),
    )

    # Process lines
    lines = [line for line in table_code.splitlines() if not line.strip().startswith("\\hline")]
    header_start = next(i for i, line in enumerate(lines) if "\\begin{tabular}" in line)

    # Create header
    header_lines = ["\\toprule"]
    if "fpr" in str(formatted_df.columns):
        header = [
            "& "
            + " & ".join(
                f"\\multicolumn{{2}}{{c}}{{\\textbf{{{metric}}} ({'$\\uparrow$' if maximize else '$\\downarrow$'})}}"
                for metric, maximize in metric_config["display"]
            )
            + " \\\\",
            "Model & "
            + " & ".join(f"{LANG_DISPLAY['english']} & {LANG_DISPLAY['italian']}" for _ in metric_config["display"])
            + " \\\\",
        ]
    else:
        header = [
            "& "
            + " & ".join(
                f"\\multicolumn{{2}}{{c}}{{\\textbf{{{metric}}} ($\\uparrow$)}}"
                for metric, _ in metric_config["display"]
            )
            + " \\\\",
            "Model & "
            + " & ".join(f"{LANG_DISPLAY['english']} & {LANG_DISPLAY['italian']}" for _ in metric_config["display"])
            + " \\\\",
        ]

    # Build table
    header_lines.extend([*header])
    data_start = header_start + 4
    data_rows = lines[data_start:]
    table_lines = lines[: header_start + 1] + header_lines + data_rows

    # Find position AFTER Llama Guard 3 8B (ICL) for the separator
    icl_model_name = "Llama Guard 3 8B (ICL)"
    icl_pos = next(i for i, line in enumerate(table_lines) if icl_model_name in line)
    table_lines.insert(icl_pos + 1, "\\addlinespace[2pt]\\midrule")

    # Generate final document
    latex_doc = (
        f"{LATEX_PREAMBLE}\n{LATEX_BEGIN}\n"
        f"\\begin{{tabular}}{{{'l' + 'c' * len(formatted_df.columns)}}}\n"
        f"{chr(10).join(table_lines[header_start + 1 :])}\n"
        "\\end{document}\n"
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(latex_doc)


def main() -> None:
    """Generate and export tables."""
    dfs = load_metrics_data(MODELS, LANGUAGES, OUTPUT_DIR)

    # Create tables
    table1 = create_metrics_table(dfs, METRICS_CONFIG["flagged"]["metrics"], include_all=INCLUDE_ALL)
    table2 = create_metrics_table(dfs, METRICS_CONFIG["average"]["metrics"], include_all=INCLUDE_ALL)

    # Export tables
    export_table_to_latex_standalone(
        table1,
        OUTPUT_DIR / "tables/table1_flagged_metrics_standalone.tex",
        METRICS_CONFIG["flagged"],
    )
    export_table_to_latex_standalone(
        table2,
        OUTPUT_DIR / "tables/table2_classification_performance_standalone.tex",
        METRICS_CONFIG["average"],
    )


if __name__ == "__main__":
    main()
