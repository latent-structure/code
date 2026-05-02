from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, percentile_interval, write_csv, write_json
from hardening_common import write_text


OUTCOMES = [
    "source_attraction",
    "source_minus_target_margin",
    "description_drift",
    "clip_source_minus_target_similarity",
    "mismatched_source_leakage",
    "rdm_disruption",
    "target_perturbation",
    "lancaster_visual_mean",
]


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.astype(float)
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=0).replace(0.0, 1.0)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    xtx = x.T @ x
    penalty = alpha * np.eye(xtx.shape[0])
    beta = np.linalg.solve(xtx + penalty, x.T @ y)
    yhat = x @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return beta, yhat, r2


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denom == 0.0 else float(np.dot(x, y) / denom)


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray, alpha: float) -> float:
    if controls.shape[1] == 0:
        return pearson(x, y)
    beta_x, _, _ = ridge_fit(controls, x, alpha)
    beta_y, _, _ = ridge_fit(controls, y, alpha)
    return pearson(x - controls @ beta_x, y - controls @ beta_y)


def bootstrap_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    moderator_names: list[str],
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in moderator_names}
    n = x.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        beta, _, _ = ridge_fit(x[idx], y[idx], alpha)
        for name, value in zip(moderator_names, beta):
            values[name].append(float(value))
    return {name: percentile_interval(np.asarray(vals, dtype=float), 0.95) for name, vals in values.items()}


def variance_inflation_factors(x: np.ndarray, names: list[str], alpha: float) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(names):
        y = x[:, idx]
        controls = np.delete(x, idx, axis=1)
        _, _, r2 = ridge_fit(controls, y, alpha)
        vif = 1.0 / max(1.0 - r2, 1e-8)
        rows.append({"moderator": name, "ridge_r2_from_other_moderators": r2, "ridge_vif": vif})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit joint THINGSplus moderator models for geometry/behavior outcomes.")
    parser.add_argument("--input", default="outputs/scope_extensions/thingsplus_concept_moderators.csv")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    df = pd.read_csv(ROOT / args.input)
    moderator_cols = [col for col in df.columns if col.startswith("thingsplus_")]
    x_df = zscore(df[moderator_cols])
    x = x_df.to_numpy(dtype=float)

    coef_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for outcome_idx, outcome in enumerate(OUTCOMES):
        if outcome not in df:
            continue
        y = pd.to_numeric(df[outcome], errors="coerce")
        mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(x).all(axis=1)
        x_sub = x[mask]
        y_sub = y.to_numpy(dtype=float)[mask]
        y_z = (y_sub - y_sub.mean()) / (y_sub.std(ddof=0) or 1.0)
        beta, yhat, r2 = ridge_fit(x_sub, y_z, args.alpha)
        ci = bootstrap_coefficients(
            x_sub,
            y_z,
            moderator_cols,
            args.alpha,
            args.bootstrap,
            args.seed + outcome_idx * 1000,
        )
        fit_rows.append(
            {
                "outcome": outcome,
                "n": int(mask.sum()),
                "ridge_alpha": args.alpha,
                "model_r2": r2,
                "predicted_observed_pearson_r": pearson(yhat, y_z),
            }
        )
        for moderator_idx, moderator in enumerate(moderator_cols):
            controls = np.delete(x_sub, moderator_idx, axis=1)
            pcorr = partial_corr(x_sub[:, moderator_idx], y_z, controls, args.alpha)
            ci_low, ci_high = ci[moderator]
            coef_rows.append(
                {
                    "outcome": outcome,
                    "moderator": moderator,
                    "n": int(mask.sum()),
                    "ridge_alpha": args.alpha,
                    "standardized_ridge_beta": float(beta[moderator_idx]),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ridge_partial_corr": pcorr,
                }
            )

    vif_rows = variance_inflation_factors(x, moderator_cols, args.alpha)
    out_dir = ROOT / "outputs" / "scope_extensions"
    report_dir = ROOT / "reports" / "main_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "thingsplus_joint_moderator_coefficients.csv",
        coef_rows,
        ["outcome", "moderator", "n", "ridge_alpha", "standardized_ridge_beta", "ci95_low", "ci95_high", "ridge_partial_corr"],
    )
    write_csv(out_dir / "thingsplus_joint_moderator_model_fits.csv", fit_rows, ["outcome", "n", "ridge_alpha", "model_r2", "predicted_observed_pearson_r"])
    write_csv(out_dir / "thingsplus_moderator_collinearity.csv", vif_rows, ["moderator", "ridge_r2_from_other_moderators", "ridge_vif"])
    top = sorted(coef_rows, key=lambda row: abs(float(row["standardized_ridge_beta"])), reverse=True)[:30]
    write_json(
        out_dir / "thingsplus_joint_moderator_summary.json",
        {
            "n_concepts": int(len(df)),
            "n_moderators": len(moderator_cols),
            "ridge_alpha": args.alpha,
            "bootstrap": args.bootstrap,
            "model_fits": fit_rows,
            "top_absolute_standardized_betas": top,
            "collinearity": vif_rows,
        },
    )
    lines = [
        "# THINGSplus Joint Moderator Models",
        "",
        f"- Concepts: `{len(df)}`",
        f"- Moderators: `{len(moderator_cols)}`",
        f"- Model: standardized ridge regression, alpha `{args.alpha}`",
        f"- Bootstrap resamples: `{args.bootstrap}`",
        "",
        "## Model Fits",
        "",
        "| Outcome | R2 | Predicted-observed r | n |",
        "|---|---:|---:|---:|",
    ]
    for row in fit_rows:
        lines.append(f"| `{row['outcome']}` | {row['model_r2']:.4f} | {row['predicted_observed_pearson_r']:.4f} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Largest Unique Moderator Coefficients",
            "",
            "| Outcome | Moderator | Std. ridge beta | 95% CI | Ridge partial r |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in top:
        lines.append(
            f"| `{row['outcome']}` | `{row['moderator']}` | {row['standardized_ridge_beta']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['ridge_partial_corr']:+.4f} |"
        )
    lines.extend(["", "## Moderator Collinearity", "", "| Moderator | Ridge VIF | R2 from other moderators |", "|---|---:|---:|"])
    for row in sorted(vif_rows, key=lambda item: float(item["ridge_vif"]), reverse=True):
        lines.append(f"| `{row['moderator']}` | {row['ridge_vif']:.2f} | {row['ridge_r2_from_other_moderators']:.4f} |")
    write_text(report_dir / "thingsplus_joint_moderator_report.md", "\n".join(lines) + "\n")
    append_run_log("THINGSplus Joint Moderators", [f"Wrote joint THINGSplus moderator models with {len(moderator_cols)} predictors."])


if __name__ == "__main__":
    main()
