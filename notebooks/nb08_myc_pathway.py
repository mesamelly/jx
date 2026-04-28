# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "polars",
#   "requests",
#   "broad-babel",
#   "biopython",
#   "matplotlib",
#   "seaborn",
#   "jump-portrait",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from broad_babel.query import get_mapper

    NOTEBOOK_DIR = Path(__file__).parent
    CACHE_DIR = Path(os.environ.get("JX_CACHE", Path.home() / ".cache" / "jx"))

    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    from nb04_display_images import display_site, lookup_site_metadata, pick_first_site
    from nb05_explore_similarity import load_distance_matrix
    from nb06_query_genes import entrez_gene_info, gene_symbols_to_ncbi, parse_gene_list


@app.function
def symbols_to_jcp(symbols: tuple[str, ...]) -> dict[str, str]:
    """Map gene symbols → JCP2022 IDs via broad-babel standard_key lookup."""
    return get_mapper(
        query=symbols,
        input_column="standard_key",
        output_columns="standard_key,JCP2022",
    )


@app.function
def load_sim_cached(dataset: str) -> pl.LazyFrame:
    """Load cosine similarity matrix, serving from local cache when available."""
    cached = CACHE_DIR / f"{dataset}_cosinesim_full.parquet"
    if cached.exists():
        return pl.scan_parquet(cached)
    return load_distance_matrix(dataset)


@app.function
def gene_submatrix(distances: pl.LazyFrame, jcp_ids: list[str]) -> pl.DataFrame:
    """Extract the square submatrix for a specific set of JCP2022 IDs."""
    all_cols = distances.collect_schema().names()
    idx = [i for i, c in enumerate(all_cols) if c in set(jcp_ids)]
    found_cols = [all_cols[i] for i in idx]
    return (
        distances.with_row_index()
        .filter(pl.col("index").is_in(idx))
        .select(pl.col(found_cols))
        .collect()
    )


@app.function
def plot_gene_heatmap(submatrix: pl.DataFrame, label_map: dict[str, str]) -> plt.Figure:
    """Seaborn heatmap of pairwise cosine similarities, labelled by gene symbol."""
    cols = submatrix.columns
    labels = [label_map.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        submatrix.to_numpy(),
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Pairwise morphological similarity (cosine, CRISPR knockouts)")
    fig.tight_layout()
    return fig


@app.cell
def intro():
    mo.md("""
    # MYC pathway gene explorer

    Enter gene symbols, click **Run analysis**, and explore their JUMP CRISPR
    morphological profiles — JCP2022 IDs, Entrez summaries, pairwise cosine
    similarity heatmap, and Cell Painting images.
    """)
    return


@app.cell
def controls():
    gene_input = mo.ui.text(
        value="MYC, MYCN, MXD1, MNT, E2F1",
        label="Gene symbols (comma-separated)",
        full_width=True,
    )
    email_input = mo.ui.text(
        value="melissali6688@gmail.com",
        label="Email for NCBI Entrez",
        full_width=True,
    )
    dataset_select = mo.ui.dropdown(
        options=["crispr", "orf"],
        value="crispr",
        label="Modality",
    )
    run_button = mo.ui.run_button(label="Run analysis")
    mo.sidebar(
        [
            mo.md("### Controls"),
            gene_input,
            email_input,
            dataset_select,
            run_button,
        ]
    )
    return dataset_select, email_input, gene_input, run_button


@app.cell
def gene_lookup(gene_input, run_button):
    mo.stop(not run_button.value)
    symbols = parse_gene_list(gene_input.value)
    jcp_map = symbols_to_jcp(symbols)
    return jcp_map, symbols


@app.cell
def jcp_table(jcp_map):
    mo.vstack([
        mo.md("## Gene → JCP2022 mapping"),
        mo.ui.table(
            pl.DataFrame({"gene": list(jcp_map.keys()), "JCP2022": list(jcp_map.values())}),
            selection=None,
        ),
    ])
    return


@app.cell
def entrez_table(email_input, symbols):
    ncbi_ids = gene_symbols_to_ncbi(symbols)
    info = entrez_gene_info(tuple(ncbi_ids.values()), email_input.value)
    mo.vstack([
        mo.md("## Entrez gene summaries"),
        mo.ui.table(info, selection=None),
    ])
    return


@app.cell
def similarity_fetch(dataset_select, jcp_map):
    sim = load_sim_cached(dataset_select.value)
    sub = gene_submatrix(sim, list(jcp_map.values()))
    return (sub,)


@app.cell
def similarity_plot(jcp_map, sub):
    inv_map = {v: k for k, v in jcp_map.items()}
    sim_fig = plot_gene_heatmap(sub, inv_map)
    mo.vstack([mo.md("## Pairwise morphological similarity"), mo.as_html(sim_fig)])
    return


@app.cell(hide_code=True)
def similarity_hist(sub):
    import numpy as np

    mat = sub.to_numpy()
    mask = ~np.eye(len(mat), dtype=bool)
    vals = mat[mask]

    hist_fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(vals, bins=30, color="#5b8db8", edgecolor="white", linewidth=0.4)
    ax.axvline(
        vals.mean(),
        color="tomato",
        linestyle="--",
        linewidth=1.2,
        label=f"mean {vals.mean():.2f}",
    )
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Off-diagonal pairwise similarities")
    ax.legend(frameon=False)
    hist_fig.tight_layout()
    mo.vstack([mo.md("## Similarity distribution"), mo.as_html(hist_fig)])
    return


@app.cell
def image_controls(jcp_map):
    gene_select = mo.ui.dropdown(
        options=jcp_map,
        label="Show Cell Painting images for gene",
    )
    mo.vstack([mo.md("## Cell Painting images"), gene_select])
    return (gene_select,)


@app.cell
def images(gene_select):
    if not gene_select.value:
        mo.stop(True)
    sites = lookup_site_metadata(gene_select.value, input_column="JCP2022")
    site = pick_first_site(sites)
    img_fig = display_site(
        site["Source"],
        site["Batch"],
        site["Plate"],
        site["Well"],
        site["Site"],
        gene_select.value,
    )
    mo.as_html(img_fig)
    return


if __name__ == "__main__":
    app.run()
