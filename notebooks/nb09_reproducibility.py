# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "polars",
#   "requests",
#   "broad-babel",
#   "numpy",
#   "plotly",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl

    NOTEBOOK_DIR = Path(__file__).parent
    CACHE_DIR = Path(os.environ.get("JX_CACHE", Path.home() / ".cache" / "jx"))

    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    from nb02_add_metadata import build_mapper

    PROFILE_INDEX_URL = "https://raw.githubusercontent.com/jump-cellpainting/datasets/v0.11.0/manifests/profile_index.json"
    FEAT_PATTERN = r"^X_\d+$"


@app.function
def fetch_wells(jcp_id: str) -> pl.DataFrame:
    """Pull all compound wells for a JCP2022 ID, using a local cache of profiles."""
    import requests

    cached = CACHE_DIR / "compound_profiles.parquet"
    if not cached.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        index = requests.get(PROFILE_INDEX_URL).json()
        url = (
            pl.DataFrame(index)
            .filter(pl.col("subset") == "compound")
            .item(0, "url")
        )
        pl.scan_parquet(url).collect().write_parquet(cached)
    return (
        pl.scan_parquet(cached)
        .filter(pl.col("Metadata_JCP2022") == jcp_id)
        .collect()
    )


@app.function
def pairwise_cosine_labeled(wells: pl.DataFrame) -> pl.DataFrame:
    """Compute all well-pair cosine similarities, labelled within- vs cross-source."""
    import polars.selectors as cs

    feat_cols = [c for c in wells.columns if c.startswith("X_")]
    feat = wells.select(feat_cols).to_numpy().astype(np.float32)
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = feat / norms
    sim = normed @ normed.T

    sources = wells["Metadata_Source"].to_list()
    n = len(sources)
    rows = [
        {
            "source_i": sources[i],
            "source_j": sources[j],
            "similarity": float(sim[i, j]),
            "pair_type": "within-source" if sources[i] == sources[j] else "cross-source",
        }
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return pl.DataFrame(rows)


@app.function
def source_pair_summary(pairs: pl.DataFrame) -> pl.DataFrame:
    """Mean and std cosine similarity for each source × source pair."""
    return (
        pairs.group_by(["source_i", "source_j", "pair_type"])
        .agg(
            pl.col("similarity").mean().alias("mean_sim"),
            pl.col("similarity").std().alias("std_sim"),
            pl.len().alias("n_pairs"),
        )
        .sort("mean_sim", descending=True)
    )


@app.cell
def intro():
    mo.md("""
    # Perturbation reproducibility across sources

    For a given compound (JCP2022 ID), fetch every well profile across all
    sources, compute pairwise cosine similarities, and compare **within-source**
    vs **cross-source** distributions. A large gap flags a perturbation whose
    morphological signal doesn't replicate.

    > **Note:** CRISPR and ORF perturbations are each measured in only one source
    > in JUMP, so this analysis only applies to compounds.
    """)
    return


@app.cell
def controls():
    jcp_input = mo.ui.text(
        value="JCP2022_091373",
        label="JCP2022 compound ID",
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Fetch & compute")
    mo.sidebar(
        [
            mo.md("### Controls"),
            jcp_input,
            mo.callout(
                mo.md(
                    "⚠ First run downloads & caches compound profiles (~1 GB). Subsequent runs are fast."
                ),
                kind="warn",
            ),
            mo.md(
                "*CRISPR/ORF perturbations are single-source and not suitable for this analysis.*"
            ),
            run_button,
        ]
    )
    return jcp_input, run_button


@app.cell
def well_fetch(jcp_input, run_button):
    mo.stop(not run_button.value)
    wells = fetch_wells(jcp_input.value)
    name_map = build_mapper((jcp_input.value,), "standard_key")
    compound_name = name_map.get(jcp_input.value, jcp_input.value)
    return compound_name, wells


@app.cell
def well_summary(compound_name, jcp_input, wells):
    per_source = (
        wells.group_by("Metadata_Source")
        .len()
        .rename({"len": "n_wells"})
        .sort("n_wells", descending=True)
    )
    mo.vstack([
        mo.md(f"## {compound_name} (`{jcp_input.value}`) — {len(wells)} wells across {per_source.height} sources"),
        mo.ui.table(per_source, selection=None),
    ])
    return


@app.cell
def pairs_compute(wells):
    pairs = pairwise_cosine_labeled(wells)
    return (pairs,)


@app.cell
def distribution_plot(compound_name, pairs):
    import plotly.graph_objects as go

    within_vals = pairs.filter(pl.col("pair_type") == "within-source")[
        "similarity"
    ].to_list()
    cross_vals = pairs.filter(pl.col("pair_type") == "cross-source")[
        "similarity"
    ].to_list()

    dist_fig = go.Figure()
    dist_fig.add_trace(
        go.Histogram(
            x=within_vals,
            name="within-source",
            opacity=0.7,
            marker_color="#5b8db8",
            nbinsx=60,
        )
    )
    dist_fig.add_trace(
        go.Histogram(
            x=cross_vals,
            name="cross-source",
            opacity=0.7,
            marker_color="#e07b54",
            nbinsx=60,
        )
    )
    dist_fig.update_layout(
        barmode="overlay",
        xaxis_title="Cosine similarity",
        yaxis_title="Count",
        title=f"Within- vs cross-source similarity — {compound_name}",
    )
    dist_fig.add_vline(
        x=float(np.mean(within_vals)),
        line_dash="dash",
        line_color="#5b8db8",
        annotation_text=f"within μ={np.mean(within_vals):.2f}",
    )
    dist_fig.add_vline(
        x=float(np.mean(cross_vals)),
        line_dash="dash",
        line_color="#e07b54",
        annotation_text=f"cross μ={np.mean(cross_vals):.2f}",
    )
    mo.vstack([mo.md("## Similarity distribution"), mo.ui.plotly(dist_fig)])
    return


@app.cell
def summary_table(pairs):
    summary = source_pair_summary(pairs)
    mo.vstack([
        mo.md("## Source-pair breakdown"),
        mo.ui.table(summary, selection=None),
    ])
    return


@app.cell
def reproducibility_flag(pairs):
    within_mean = pairs.filter(pl.col("pair_type") == "within-source")["similarity"].mean()
    cross_mean  = pairs.filter(pl.col("pair_type") == "cross-source")["similarity"].mean()
    delta = within_mean - cross_mean if within_mean and cross_mean else None

    if delta is None:
        verdict = mo.callout(mo.md("Not enough data to assess reproducibility."), kind="warn")
    elif delta < 0.05:
        verdict = mo.callout(mo.md(f"**High reproducibility** — within/cross-source gap = {delta:.3f}"), kind="success")
    elif delta < 0.20:
        verdict = mo.callout(mo.md(f"**Moderate reproducibility** — within/cross-source gap = {delta:.3f}"), kind="warn")
    else:
        verdict = mo.callout(mo.md(f"**Low reproducibility** — within/cross-source gap = {delta:.3f}. Treat hits with caution."), kind="danger")

    verdict
    return


if __name__ == "__main__":
    app.run()
