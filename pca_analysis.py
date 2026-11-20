"""Principal Component Analysis of yield curve changes.

Uses eigendecomposition of the covariance matrix (no sklearn dependency).
Analyzes daily yield changes for level, slope, and curvature factors.
"""

import numpy as np
import pandas as pd
from data_pipeline import DEFAULT_WINDOW_END, DEFAULT_WINDOW_START, PCA_TENORS, TENOR_MAP, scenario_changes


def prepare_pca_data(
    df_yields: pd.DataFrame,
    tenors: list[str] | None = None,
    start_date: str = DEFAULT_WINDOW_START,
    end_date: str = DEFAULT_WINDOW_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare centered daily yield changes for PCA.

    Returns:
        changes_bp: Daily changes in basis points (with date column)
        centered: Centered changes (mean-subtracted) without date column
    """
    if tenors is None:
        tenors = PCA_TENORS

    changes = scenario_changes(df_yields, tenors, start_date, end_date)

    # Center the data
    centered = changes[tenors] - changes[tenors].mean()

    return changes, centered


def run_pca(
    df_yields: pd.DataFrame,
    tenors: list[str] | None = None,
    n_components: int = 3,
    start_date: str = DEFAULT_WINDOW_START,
    end_date: str = DEFAULT_WINDOW_END,
) -> dict:
    """Run PCA on daily yield changes using covariance matrix eigendecomposition.

    Returns dict with:
        eigenvalues, eigenvectors, explained_variance_ratio,
        cumulative_variance, loadings, scores, tenors, changes_df
    """
    if tenors is None:
        tenors = PCA_TENORS

    changes_df, centered = prepare_pca_data(df_yields, tenors, start_date, end_date)
    data_matrix = centered.values

    if n_components < 1 or len(data_matrix) < 2:
        raise ValueError("PCA requires a positive component count and at least two aligned changes.")
    cov_matrix = np.atleast_2d(np.cov(data_matrix, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[idx], 0)
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    if total_var <= 0:
        raise ValueError("PCA requires nonzero yield-change variance.")
    explained_ratio = eigenvalues / total_var
    cumulative = np.cumsum(explained_ratio)

    # Select top components
    n = min(n_components, len(eigenvalues))
    loadings = eigenvectors[:, :n]
    # Eigenvector signs are arbitrary. Give each largest loading a positive sign.
    dominant = np.argmax(np.abs(loadings), axis=0)
    loadings *= np.sign(loadings[dominant, np.arange(n)])
    scores = data_matrix @ loadings

    # Build loadings DataFrame
    tenor_years = [TENOR_MAP[t] for t in tenors]
    loadings_df = pd.DataFrame(
        loadings,
        index=tenors,
        columns=[f"PC{i+1}" for i in range(n)],
    )
    loadings_df["tenor_years"] = tenor_years

    # Build scores DataFrame
    scores_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(n)],
    )
    scores_df["date"] = changes_df["date"].values

    return {
        "eigenvalues": eigenvalues[:n],
        "explained_variance_ratio": explained_ratio[:n],
        "cumulative_variance": cumulative[:n],
        "loadings_df": loadings_df,
        "scores_df": scores_df,
        "tenors": tenors,
        "tenor_years": tenor_years,
        "n_observations": len(data_matrix),
        "cov_matrix": cov_matrix,
        "changes_df": changes_df,
    }


def interpret_components(loadings_df: pd.DataFrame) -> list[str]:
    """Heuristic interpretation of first three principal components."""
    interpretations = []
    for i, col in enumerate(["PC1", "PC2", "PC3"]):
        if col not in loadings_df.columns:
            break
        loads = loadings_df[col].values
        sign_consistency = np.all(loads > 0) or np.all(loads < 0)

        if sign_consistency:
            interpretations.append(
                f"PC{i+1}: Loadings have same sign across maturities — "
                f"consistent with a LEVEL shift (parallel movement)."
            )
        elif i == 1:
            # Check for monotonic pattern (slope)
            if (loads[0] > 0 and loads[-1] < 0) or (loads[0] < 0 and loads[-1] > 0):
                interpretations.append(
                    f"PC{i+1}: Loadings change sign from short to long maturities — "
                    f"consistent with a SLOPE factor (steepening/flattening)."
                )
            else:
                interpretations.append(
                    f"PC{i+1}: Mixed-sign loadings — examine the pattern to determine "
                    f"whether this represents slope or another structural shift."
                )
        elif i == 2 and len(loads) >= 3:
            # Check for curvature (belly vs wings)
            mid = len(loads) // 2
            belly = np.mean(loads[(len(loads) - 1) // 2:mid + 1])
            if loads[0] * belly < 0 and loads[-1] * belly < 0:
                interpretations.append(
                    f"PC{i+1}: Loadings show opposite signs at extremes vs middle — "
                    f"consistent with a CURVATURE factor (butterfly)."
                )
            else:
                interpretations.append(
                    f"PC{i+1}: Third component — inspect loadings to determine "
                    f"whether this resembles curvature or another pattern."
                )
    return interpretations


def pca_factor_shocks(pca_results: dict, factor_idx: int,
                      shock_std: float = 1.0) -> dict:
    """Generate yield curve shock from a PCA factor movement.

    Args:
        pca_results: Output from run_pca()
        factor_idx: 0-based index (0=PC1, 1=PC2, 2=PC3)
        shock_std: Number of standard deviations to shock

    Returns:
        Dict of {tenor: shock_in_bp}
    """
    std = np.sqrt(pca_results["eigenvalues"][factor_idx])
    loadings = pca_results["loadings_df"][f"PC{factor_idx+1}"].values
    shocks = loadings * std * shock_std  # in basis points
    return dict(zip(pca_results["tenors"], shocks))