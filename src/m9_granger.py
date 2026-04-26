"""
M9 — GRANGER CAUSALITY TESTS
================================
Input:  data/processed/sentiment_features.csv + data/external/vf_inflation_biweekly.csv
Output: outputs/granger_results.json + outputs/granger_plots/

Prasyarat:
  - Stationarity check (ADF + KPSS)
  - Optimal lag selection (AIC)
  - 5 Granger test pairs
  - Cross-correlation analysis
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m9")
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Stationarity Tests ───────────────────────────────────────

def stationarity_check(series: pd.Series, name: str) -> dict:
    """
    Test stationarity using ADF and KPSS.
    ADF: H0 = unit root (non-stationary). Reject H0 -> stationary.
    KPSS: H0 = stationary. Reject H0 -> non-stationary.
    """
    clean = series.dropna()
    if len(clean) < 10:
        logger.warning(f"Series '{name}' too short ({len(clean)} obs) for stationarity test")
        return {
            "variable": name,
            "adf_pvalue": None,
            "kpss_pvalue": None,
            "is_stationary": None,
            "action_needed": "insufficient_data",
        }

    # ADF test
    try:
        adf_result = adfuller(clean, autolag="AIC")
        adf_stat, adf_pval = adf_result[0], adf_result[1]
    except Exception as e:
        logger.warning(f"ADF test failed for {name}: {e}")
        adf_stat, adf_pval = None, None

    # KPSS test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(clean, regression="c", nlags="auto")
        kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]
    except Exception as e:
        logger.warning(f"KPSS test failed for {name}: {e}")
        kpss_stat, kpss_pval = None, None

    # Decision
    is_stationary = None
    if adf_pval is not None and kpss_pval is not None:
        is_stationary = (adf_pval < 0.05) and (kpss_pval > 0.05)

    result = {
        "variable": name,
        "adf_statistic": round(float(adf_stat), 4) if adf_stat is not None else None,
        "adf_pvalue": round(float(adf_pval), 4) if adf_pval is not None else None,
        "kpss_statistic": round(float(kpss_stat), 4) if kpss_stat is not None else None,
        "kpss_pvalue": round(float(kpss_pval), 4) if kpss_pval is not None else None,
        "is_stationary": is_stationary,
        "action_needed": "none" if is_stationary else "first_difference",
    }

    logger.info(
        f"Stationarity [{name}]: ADF p={adf_pval}, KPSS p={kpss_pval} "
        f"-> {'STATIONARY' if is_stationary else 'NON-STATIONARY'}"
    )

    return result


def make_stationary(series: pd.Series, name: str) -> tuple:
    """
    Apply first differencing if series is non-stationary.
    Returns: (transformed_series, transformation_applied)
    """
    result = stationarity_check(series, name)

    if result["is_stationary"]:
        return series, "none"

    # First difference
    diff_series = series.diff().dropna()
    diff_result = stationarity_check(diff_series, f"{name}_diff1")

    if diff_result["is_stationary"]:
        logger.info(f"First difference made {name} stationary")
        return diff_series, "first_difference"
    else:
        # Second difference
        diff2_series = diff_series.diff().dropna()
        logger.warning(f"Second difference applied to {name}")
        return diff2_series, "second_difference"


# ── Lag Selection ────────────────────────────────────────────

def select_optimal_lag(y: np.ndarray, x: np.ndarray, max_lag: int = 4,
                        ic: str = "aic") -> int:
    """
    Select optimal lag using VAR and information criterion.
    Handles size mismatch (e.g. after differencing one series but not the other)
    by truncating both arrays to the shorter length.
    """
    # Align lengths — first-differencing reduces length by 1
    min_len = min(len(y), len(x))
    y = y[-min_len:]
    x = x[-min_len:]
    data = np.column_stack([y, x])

    try:
        model = VAR(data)
        results = model.select_order(maxlags=max_lag)

        ic_values = {
            "aic": results.aic,
            "bic": results.bic,
            "hqic": results.hqic,
        }

        optimal_lag = ic_values.get(ic, results.aic)
        logger.info(f"Optimal lag ({ic}): {optimal_lag}")
        return max(1, min(optimal_lag, max_lag))

    except Exception as e:
        logger.warning(f"VAR lag selection failed: {e}. Using default lag=2.")
        return 2


# ── Granger Causality Test ───────────────────────────────────

GRANGER_PAIRS = [
    ("sentiment_all", "Semua berita -> Inflasi VF (agregat)"),
    ("sentiment_typeA", "Tipe A supply-driven -> Inflasi VF (STAR)"),
    ("sentiment_typeB", "Tipe B price-driven -> Inflasi VF (harusnya lemah)"),
    ("sentiment_supply", "SUPPLYSHOCK -> Inflasi VF"),
    ("sentiment_demand", "DEMANDSHOCK -> Inflasi VF"),
]


def granger_test_pair(inflation: pd.Series, predictor: pd.Series,
                       max_lag: int, alpha: float) -> dict:
    """
    Test Granger causality: does predictor help predict inflation?
    H0: predictor does NOT Granger-cause inflation.
    Reject H0 if p < alpha -> predictor IS Granger cause.
    """
    data = pd.concat([inflation, predictor], axis=1).dropna()

    if len(data) < max_lag + 5:
        logger.warning(f"Insufficient data ({len(data)} obs) for Granger test with max_lag={max_lag}")
        return {"error": "insufficient_data"}

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        output = {}
        for lag in range(1, max_lag + 1):
            test_results = results[lag][0]
            f_test = test_results["ssr_ftest"]
            f_stat = f_test[0]
            p_val = f_test[1]

            output[f"lag_{lag}"] = {
                "f_statistic": round(float(f_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant": p_val < alpha,
            }

        return output

    except Exception as e:
        logger.error(f"Granger test failed: {e}")
        return {"error": str(e)}


# ── Cross-Correlation Analysis ───────────────────────────────

def cross_correlation_plot(inflation: pd.Series, predictor: pd.Series,
                            max_lag: int = 6, name: str = "") -> None:
    """
    Plot cross-correlation for lag -max_lag to +max_lag.
    Peak at negative lag = predictor leads inflation.
    """
    lags = list(range(-max_lag, max_lag + 1))
    ccf_values = []

    for lag in lags:
        # CCF(lag) = corr(inflation_t, predictor_{t-lag})
        # Positive lag = predictor leads (shifted forward in time)
        # Negative lag = inflation leads
        shifted = predictor.shift(lag)

        valid = pd.concat([inflation, shifted], axis=1).dropna()
        if len(valid) > 2:
            ccf_values.append(inflation.corr(shifted))
        else:
            ccf_values.append(0.0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['#e74c3c' if l < 0 else '#3498db' if l > 0 else '#2c3e50' for l in lags]
    bars = ax.bar(lags, ccf_values, color=colors, alpha=0.7, edgecolor="white")

    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)

    # Significance lines (approximate 95% CI)
    n = len(inflation.dropna())
    if n > 0:
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color="gray", linestyle=":", alpha=0.5, label=f"95% CI ({ci:.3f})")
        ax.axhline(-ci, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Lag (bi-minggu negatif = predictor mendahului)")
    ax.set_ylabel("Korelasi")
    ax.set_title(f"Cross-Correlation: {name} vs Inflasi VF")
    ax.legend()

    save_name = name.replace(" ", "_").lower()
    plot_path = ROOT_DIR / "outputs" / "granger_plots" / f"ccf_{save_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"CCF plot saved: {plot_path}")


# ── Main Pipeline ────────────────────────────────────────────

def run_granger_tests(cfg: dict = None) -> dict:
    """Run the full Granger causality test pipeline."""
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    sentiment_path = ROOT_DIR / "data" / "processed" / "sentiment_features.csv"
    inflation_path = ROOT_DIR / "data" / "external" / "vf_inflation_biweekly.csv"
    output_path = ROOT_DIR / "outputs" / "granger_results.json"
    plots_dir = ROOT_DIR / "outputs" / "granger_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load sentiment features
    if not sentiment_path.exists():
        logger.error(f"Sentiment features not found: {sentiment_path}. Run m8_aggregate.py first.")
        return {}

    sentiment_df = pd.read_csv(sentiment_path)
    logger.info(f"Loaded sentiment features: {sentiment_df.shape}")

    # Load inflation data
    if not inflation_path.exists():
        logger.warning(
            f"Inflation data not found: {inflation_path}. "
            "Creating synthetic placeholder for testing. "
            "Replace with actual vf_inflation_biweekly.csv before final analysis!"
        )
        # Create synthetic placeholder
        np.random.seed(42)
        n_periods = len(sentiment_df)
        inflation_df = pd.DataFrame({
            "period_id": sentiment_df["period_id"],
            "vf_inflation": np.random.normal(0.05, 0.02, n_periods).cumsum(),
        })
        inflation_df.to_csv(inflation_path, index=False)
        logger.info(f"Synthetic inflation data saved: {inflation_path}")
    else:
        inflation_df = pd.read_csv(inflation_path)

    logger.info(f"Loaded inflation data: {inflation_df.shape}")

    # Merge
    merged = sentiment_df.merge(inflation_df, on="period_id", how="inner")
    logger.info(f"Merged dataset: {merged.shape}")

    if len(merged) < 20:
        logger.error(f"Insufficient data ({len(merged)} periods) for meaningful Granger tests.")
        return {}

    # Configuration
    max_lag = cfg["granger"]["max_lag_periods"]
    alpha = cfg["granger"]["significance_level"]
    ic = cfg["granger"]["ic_criterion"]

    # Step 1: Stationarity checks
    stationarity_results = {}
    transformations = {}

    # Check inflation
    inflation_series = merged["vf_inflation"]
    stat_result = stationarity_check(inflation_series, "vf_inflation")
    stationarity_results["vf_inflation"] = stat_result
    if not stat_result.get("is_stationary", True):
        inflation_series, trans = make_stationary(inflation_series, "vf_inflation")
        transformations["vf_inflation"] = trans
    else:
        transformations["vf_inflation"] = "none"

    # Check each sentiment variable
    for col, label in GRANGER_PAIRS:
        if col in merged.columns:
            stat_result = stationarity_check(merged[col], col)
            stationarity_results[col] = stat_result

    # Step 2: Optimal lag selection
    if "sentiment_typeA" in merged.columns:
        optimal_lag = select_optimal_lag(
            inflation_series.values,
            merged["sentiment_typeA"].values,
            max_lag=max_lag,
            ic=ic,
        )
    else:
        optimal_lag = 2

    # Step 3: Run Granger tests for all pairs
    granger_results = {}
    for col, desc in GRANGER_PAIRS:
        if col not in merged.columns:
            logger.warning(f"Column {col} not found. Skipping.")
            continue

        predictor = merged[col].copy()

        # Make stationary if needed
        stat = stationarity_results.get(col, {})
        if stat.get("is_stationary") is False:
            predictor, trans = make_stationary(predictor, col)
            transformations[col] = trans
            # Align with inflation (same length after differencing)
            min_len = min(len(inflation_series), len(predictor))
            test_inflation = inflation_series.iloc[-min_len:].reset_index(drop=True)
            test_predictor = predictor.iloc[-min_len:].reset_index(drop=True)
        else:
            transformations[col] = "none"
            # Align: if inflation was first-differenced (starts from period 2),
            # drop the first observation of the predictor to match
            if transformations.get("vf_inflation", "none") != "none":
                test_inflation = inflation_series.reset_index(drop=True)
                test_predictor = predictor.iloc[1:].reset_index(drop=True)
            else:
                test_inflation = inflation_series.reset_index(drop=True)
                test_predictor = predictor.reset_index(drop=True)

        logger.info(f"\nGranger test: {desc}")
        result = granger_test_pair(
            test_inflation.rename("vf_inflation"),
            test_predictor.rename(col),
            max_lag=max_lag,
            alpha=alpha,
        )
        granger_results[col] = result

        # Log results
        if "error" not in result:
            for lag_key, lag_result in result.items():
                sig = "***" if lag_result["significant"] else ""
                logger.info(
                    f"  {lag_key}: F={lag_result['f_statistic']:.4f}, "
                    f"p={lag_result['p_value']:.4f} {sig}"
                )

        # Step 4: Cross-correlation plot
        try:
            cross_correlation_plot(
                test_inflation, test_predictor,
                max_lag=6, name=col,
            )
        except Exception as e:
            logger.warning(f"CCF plot failed for {col}: {e}")

    # Compile final results
    final_results = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "n_periods": len(merged),
        "stationarity": {
            k: {kk: vv for kk, vv in v.items() if kk != "variable"}
            for k, v in stationarity_results.items()
        },
        "transformations": transformations,
        "optimal_lag": optimal_lag,
        "max_lag_tested": max_lag,
        "significance_level": alpha,
        "results": granger_results,
    }

    # Generate interpretation
    typeA = granger_results.get("sentiment_typeA", {})
    typeB = granger_results.get("sentiment_typeB", {})

    if "error" not in typeA and "error" not in typeB:
        typeA_sig = any(v.get("significant", False) for v in typeA.values())
        typeB_sig = any(v.get("significant", False) for v in typeB.values())

        if typeA_sig and not typeB_sig:
            interpretation = (
                "Tipe A Granger-causes inflasi VF (p<0.05), konfirmasi nilai prediktif "
                "konten supply-driven. Tipe B tidak signifikan — dekomposisi A/B tervalidasi."
            )
        elif typeA_sig and typeB_sig:
            interpretation = (
                "Kedua Tipe A dan B signifikan. Tipe A tetap kandidat utama prediktor "
                "forward-looking, tapi Tipe B juga mengandung informasi prediktif."
            )
        elif not typeA_sig and typeB_sig:
            interpretation = (
                "Tipe A tidak signifikan sebagai prediktor independen. "
                "Tipe B (price-driven) justru Granger-causes inflasi VF — konsisten dengan "
                "hipotesis media attention effect: liputan harga intensif mendorong ekspektasi "
                "inflasi dan perilaku hoarding, bukan mencerminkan supply shock yang mendasarinya. "
                "Berlawanan dengan Kwon et al. (2025) di pasar AS yang lebih efisien."
            )
        elif not typeA_sig and not typeB_sig:
            interpretation = (
                "Tipe A dan B keduanya tidak signifikan. Null finding yang valid — "
                "nilai prediktif marginal sentimen berita tidak terbukti untuk konteks ini. "
                "Rekomendasikan BSTS tanpa komponen sentimen (Model M2 saja)."
            )

        final_results["interpretation"] = interpretation

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nGranger results saved: {output_path}")
    if "interpretation" in final_results:
        logger.info(f"Interpretation: {final_results['interpretation']}")

    return final_results


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    results = run_granger_tests(cfg)
    logger.info("M9 Granger tests complete.")
