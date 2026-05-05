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

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    NOTEBOOK_DIR = Path(__file__).parent
    CACHE_DIR = Path(os.environ.get("JX_CACHE", Path.home() / ".cache" / "jx"))

    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    from nb02_add_metadata import build_mapper
    from nb09_reproducibility import fetch_wells


@app.function
def pairwise_cosine_by_plate(wells: pl.DataFrame) -> pl.DataFrame:
    """Pairwise cosine similarities labelled within-plate vs cross-plate."""
    feat_cols = [c for c in wells.columns if c.startswith("X_")]
    feat = wells.select(feat_cols).to_numpy().astype(np.float32)
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    normed = feat / np.where(norms == 0, 1.0, norms)
    sim = normed @ normed.T

    plates = wells["Metadata_Plate"].to_list()
    n = len(plates)
    rows = [
        {
            "plate_i": plates[i],
            "plate_j": plates[j],
            "similarity": float(sim[i, j]),
            "pair_type": "within-plate" if plates[i] == plates[j] else "cross-plate",
        }
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return pl.DataFrame(rows)


@app.function
def plate_pair_summary(pairs: pl.DataFrame) -> pl.DataFrame:
    """Mean cosine similarity for each plate × plate pair."""
    return (
        pairs.group_by(["plate_i", "plate_j", "pair_type"])
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
    # Within-source batch reproducibility

    For a compound measured on multiple plates within a single source, compare
    **within-plate** vs **cross-plate** cosine similarities. Each plate is a
    separate imaging run — a proxy for batch. A large gap means the signal
    varies more between runs than within them.

    > Batch identity is inferred from `Metadata_Plate` — there is no explicit
    > `Metadata_Batch` column in the JUMP compound profiles.
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
    mo.sidebar([
        mo.md("### Controls"),
        jcp_input,
        run_button,
    ])
    return jcp_input, run_button


@app.cell
def well_fetch(jcp_input, run_button):
    mo.stop(not run_button.value)
    all_wells = fetch_wells(jcp_input.value)
    name_map = build_mapper((jcp_input.value,), "standard_key")
    compound_name = name_map.get(jcp_input.value, jcp_input.value)
    sources = sorted(all_wells["Metadata_Source"].unique().to_list())
    return all_wells, compound_name, sources


@app.cell
def source_selector(sources):
    source_select = mo.ui.dropdown(
        options=sources,
        value=sources[0] if sources else None,
        label="Source to analyse",
    )
    mo.vstack([mo.md("## Select source"), source_select])
    return (source_select,)


@app.cell
def source_wells(all_wells, compound_name, jcp_input, source_select):
    wells = all_wells.filter(pl.col("Metadata_Source") == source_select.value)
    n_plates = wells["Metadata_Plate"].n_unique()
    mo.vstack([
        mo.md(f"## {compound_name} (`{jcp_input.value}`) in {source_select.value} — {len(wells)} wells across {n_plates} plates"),
        mo.ui.table(
            wells.group_by("Metadata_Plate").len().rename({"len": "n_wells"}).sort("Metadata_Plate"),
            selection=None,
        ),
    ])
    return (wells,)


@app.cell
def pairs_compute(wells):
    mo.stop(wells["Metadata_Plate"].n_unique() < 2,
            mo.callout(mo.md("Need at least 2 plates for batch comparison."), kind="warn"))
    pairs = pairwise_cosine_by_plate(wells)
    return (pairs,)


@app.cell
def distribution_plot(compound_name, pairs, source_select):
    within = pairs.filter(pl.col("pair_type") == "within-plate")["similarity"].to_list()
    cross  = pairs.filter(pl.col("pair_type") == "cross-plate")["similarity"].to_list()

    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(x=within, name="within-plate", opacity=0.7, marker_color="#5b8db8", nbinsx=40))
    dist_fig.add_trace(go.Histogram(x=cross,  name="cross-plate",  opacity=0.7, marker_color="#e07b54", nbinsx=40))
    dist_fig.update_layout(
        barmode="overlay",
        xaxis_title="Cosine similarity",
        yaxis_title="Count",
        title=f"Plate-level reproducibility — {compound_name} in {source_select.value}",
    )
    if within:
        dist_fig.add_vline(x=float(np.mean(within)), line_dash="dash", line_color="#5b8db8",
                           annotation_text=f"within μ={np.mean(within):.2f}")
    if cross:
        dist_fig.add_vline(x=float(np.mean(cross)), line_dash="dash", line_color="#e07b54",
                           annotation_text=f"cross μ={np.mean(cross):.2f}")
    mo.vstack([mo.md("## Similarity distribution"), mo.ui.plotly(dist_fig)])
    return


@app.cell
def summary_table(pairs):
    summary = plate_pair_summary(pairs)
    mo.vstack([
        mo.md("## Plate-pair breakdown"),
        mo.ui.table(summary, selection=None),
    ])
    return


@app.cell
def reproducibility_flag(pairs):
    within_mean = pairs.filter(pl.col("pair_type") == "within-plate")["similarity"].mean()
    cross_mean  = pairs.filter(pl.col("pair_type") == "cross-plate")["similarity"].mean()
    delta = (within_mean - cross_mean) if (within_mean is not None and cross_mean is not None) else None

    if delta is None:
        verdict = mo.callout(mo.md("Not enough data to assess batch reproducibility."), kind="warn")
    elif delta < 0.05:
        verdict = mo.callout(mo.md(f"**High batch reproducibility** — within/cross-plate gap = {delta:.3f}"), kind="success")
    elif delta < 0.20:
        verdict = mo.callout(mo.md(f"**Moderate batch reproducibility** — within/cross-plate gap = {delta:.3f}"), kind="warn")
    else:
        verdict = mo.callout(mo.md(f"**Low batch reproducibility** — within/cross-plate gap = {delta:.3f}. Possible plate effects."), kind="danger")

    verdict
    return


if __name__ == "__main__":
    app.run()
