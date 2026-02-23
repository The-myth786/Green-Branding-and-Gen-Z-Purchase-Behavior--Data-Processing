# =============================================================================
# GREEN BRANDING & GEN Z PURCHASE BEHAVIOR — FULL STATISTICAL ANALYSIS PIPELINE
# Study Title: "Green Branding and Gen Z Purchase Behavior: A Multi-Level Empirical Analysis"
# Author      : [Your Name]
# Data Source : Primary Survey (Google Forms export)
# Python      : 3.9+
#
# REQUIRED PACKAGES (install via pip if missing):
#   pip install pandas numpy statsmodels seaborn matplotlib scipy openpyxl
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATH = "Green_Branding_and_Buying_Decisions_Among_Generation_Z___Responses_.xlsx"

# =============================================================================
# SECTION 0 ── CONSTRUCT / COLUMN MAPPING
# =============================================================================
# Maps construct names to the exact survey column headers.
# Items marked _R are REVERSE-CODED (scored 6 − raw value).
#
# Environmental Identity  — measures how central eco-values are to self-concept
# Green Branding          — measures perception of green marketing claims
# Social Influence        — measures peer/influencer effects on green behaviour
# Price Sensitivity       — measures cost as a barrier to sustainable purchase
# Purchase Intention (DV) — binary: Yes = 1, No = 0

LIKERT_COLS = {
    # ── Environmental Identity (3 items) ──────────────────────────────────────
    "EI1": "Sustainability is an important part of who I am.",
    "EI2": "I feel responsible for reducing environmental harm.",
    "EI3": "I actively try to make environmentally friendly choices.",

    # ── Green Branding Perception (5 items; GB5 is reverse-coded) ────────────
    "GB1": "I notice eco-labels and green packaging while shopping.",
    "GB2": "Green branding increases my trust in a company.",
    "GB3": "Sustainable brands appear more premium to me.",
    "GB4": "I believe most brands\u2019 environmental claims are genuine.",   # Unicode right-single-quote
    "GB5_R": "Many brands exaggerate their green claims. ",   # trailing space in original

    # ── Social Influence (2 items) ────────────────────────────────────────────
    "SI1": "Social media influencers shape my view of sustainable brands.",
    "SI2": "If a sustainable product trends online, I\u2019m more likely to try it.",

    # ── Price Sensitivity (2 items; PS1 is reverse-coded) ────────────────────
    # PS1: high willingness to pay → LOW price sensitivity → reverse for construct
    "PS1_R": "I am willing to pay more for eco-friendly products.",
    "PS2":   "Price is more important than sustainability when I buy products. ",
}

REVERSE_ITEMS = ["GB5_R", "PS1_R"]   # items that need 6 − score

CONSTRUCTS = {
    "Environmental_Identity": ["EI1", "EI2", "EI3"],
    "Green_Branding":         ["GB1", "GB2", "GB3", "GB4", "GB5_R"],
    "Social_Influence":       ["SI1", "SI2"],
    "Price_Sensitivity":      ["PS1_R", "PS2"],
}

DV_COL  = "I intend to increase my purchase of sustainable products in the future. "  # trailing space matches raw Excel column
AGE_COL = "Age:"

# Likert text → numeric mapping (case-insensitive after normalisation)
LIKERT_MAP = {
    "strongly disagree": 1,
    "disagree":          2,
    "neutral":           3,
    "agree":             4,
    "strongly agree":    5,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the Excel dataset and return a raw DataFrame."""
    print(f"\n{'='*70}")
    print("STEP 1 ── LOADING DATASET")
    print(f"{'='*70}")
    df = pd.read_excel("Green Branding and Buying Decisions Among Generation Z  (Responses).xlsx")
    print(f"  Raw shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns   : {list(df.columns)}\n")
    print(df.info())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    return df


def clean_age(df: pd.DataFrame, col: str = AGE_COL,
              min_age: int = 16, max_age: int = 26) -> pd.DataFrame:
    """
    Coerce age to numeric, drop non-parseable values,
    and keep only respondents aged min_age–max_age (Gen Z range).
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[col])
    df = df[(df[col] >= min_age) & (df[col] <= max_age)]
    after = len(df)
    print(f"\n  Age filter: kept {after} / {before} rows (ages {min_age}–{max_age})")
    return df.reset_index(drop=True)


def convert_likert(df: pd.DataFrame,
                   col_map: dict,
                   reverse_items: list) -> pd.DataFrame:
    """
    Convert Likert text responses to numeric 1–5.
    Reverse-coded items are transformed as: 6 − score.

    Interpretation note:
      1 = Strongly Disagree … 5 = Strongly Agree
      For reverse items higher numeric = stronger DISagreement with the
      positive direction of the construct.
    """
    df = df.copy()
    for key, original_col in col_map.items():
        # Normalise casing/whitespace before mapping
        series = df[original_col].astype(str).str.strip().str.lower()
        df[key] = series.map(LIKERT_MAP)

        if key in reverse_items:
            # Reverse-score: 1→5, 2→4, 3→3, 4→2, 5→1
            df[key] = 6 - df[key]

    return df


def encode_dv(df: pd.DataFrame, dv_col: str = DV_COL) -> pd.DataFrame:
    """
    Encode the binary Purchase Intention DV:
      Yes → 1  |  No → 0
    Note: Because the DV is dichotomous, the MLR below is a
    Linear Probability Model (LPM). A supplementary Logistic
    Regression is also provided for robustness.
    """
    df = df.copy()
    df["Purchase_Intention"] = (
        df[dv_col].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0})
    )
    return df


def drop_missing(df: pd.DataFrame, subset_cols: list) -> pd.DataFrame:
    """Remove rows with any NaN in the analysis columns."""
    before = len(df)
    df = df.dropna(subset=subset_cols).reset_index(drop=True)
    after = len(df)
    print(f"\n  Missing-value removal: {before - after} rows dropped → {after} rows remain")
    return df


def make_composites(df: pd.DataFrame, constructs: dict) -> pd.DataFrame:
    """
    Create composite construct scores by averaging their item scores.
    Averaging (rather than summing) keeps the scale interpretable
    on the original 1–5 Likert metric.
    """
    df = df.copy()
    for construct, items in constructs.items():
        df[construct] = df[items].mean(axis=1)
    return df


# ── Reliability ───────────────────────────────────────────────────────────────

def cronbach_alpha(item_df: pd.DataFrame) -> float:
    """
    Compute Cronbach's Alpha for a set of items.

    Interpretation:
      α ≥ 0.90  → Excellent
      α ≥ 0.80  → Good
      α ≥ 0.70  → Acceptable  (minimum threshold for research)
      α ≥ 0.60  → Questionable
      α < 0.60  → Poor (reconsider the scale)

    Formula:
      α = (k / (k−1)) × (1 − Σvarᵢ / varₜₒₜₐₗ)
    """
    items = item_df.dropna()
    k = items.shape[1]
    if k < 2:
        return np.nan
    item_vars  = items.var(axis=0, ddof=1).sum()
    total_var  = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_vars / total_var)
    return round(alpha, 4)


def reliability_report(df: pd.DataFrame, constructs: dict) -> pd.DataFrame:
    """
    Compute and display Cronbach's Alpha for each construct.
    Returns a summary DataFrame.
    """
    print(f"\n{'='*70}")
    print("STEP 5 ── CRONBACH'S ALPHA (INTERNAL CONSISTENCY RELIABILITY)")
    print(f"{'='*70}")
    print("  Threshold: α ≥ 0.70 is acceptable for social-science research.")
    print(f"  {'Construct':<30} {'Items':>5} {'Alpha':>8}  {'Verdict'}")
    print(f"  {'-'*60}")

    rows = []
    for construct, items in constructs.items():
        alpha = cronbach_alpha(df[items])
        if alpha >= 0.90:
            verdict = "Excellent"
        elif alpha >= 0.80:
            verdict = "Good"
        elif alpha >= 0.70:
            verdict = "Acceptable"
        elif alpha >= 0.60:
            verdict = "Questionable"
        else:
            verdict = "Poor"
        print(f"  {construct:<30} {len(items):>5} {alpha:>8.4f}  {verdict}")
        rows.append({"Construct": construct, "N_Items": len(items),
                     "Cronbach_Alpha": alpha, "Verdict": verdict})

    return pd.DataFrame(rows)


# ── Descriptive Statistics ────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, construct_names: list) -> pd.DataFrame:
    """
    Generate descriptive statistics table (N, Mean, SD, Min, Max, Skewness,
    Kurtosis) for each composite construct and the DV.
    """
    print(f"\n{'='*70}")
    print("STEP 6 ── DESCRIPTIVE STATISTICS")
    print(f"{'='*70}")

    cols  = construct_names + ["Purchase_Intention"]
    desc  = df[cols].agg(["count", "mean", "std", "min", "max",
                           lambda x: x.skew(), lambda x: x.kurt()])
    desc.index = ["N", "Mean", "SD", "Min", "Max", "Skewness", "Kurtosis"]
    desc = desc.round(4)
    print(desc.to_string())
    return desc


# ── Correlation Matrix ────────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame,
                        construct_names: list,
                        save_path: str) -> pd.DataFrame:
    """
    Compute Pearson correlations among constructs + DV,
    annotate statistical significance, and save a heatmap.

    Interpretation:
      |r| < 0.20   → Negligible
      0.20–0.39    → Weak
      0.40–0.59    → Moderate
      0.60–0.79    → Strong
      |r| ≥ 0.80   → Very strong
    """
    print(f"\n{'='*70}")
    print("STEP 7 ── CORRELATION MATRIX")
    print(f"{'='*70}")

    cols   = construct_names + ["Purchase_Intention"]
    corr   = df[cols].corr(method="pearson")
    pvals  = pd.DataFrame(np.ones_like(corr), columns=corr.columns, index=corr.index)

    for i in cols:
        for j in cols:
            if i != j:
                r, p = stats.pearsonr(df[i].dropna(), df[j].dropna())
                pvals.loc[i, j] = p

    print("\nPearson Correlation Matrix:")
    print(corr.round(3).to_string())
    print("\nTwo-tailed p-values:")
    print(pvals.round(4).to_string())

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    mask    = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, square=True, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "Pearson r"}
    )
    ax.set_title("Pearson Correlation Matrix\n(Lower Triangle)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Heatmap saved → {save_path}")

    return corr


# ── Multiple Linear Regression ────────────────────────────────────────────────

def run_mlr(df: pd.DataFrame,
            dv: str,
            ivs: list) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run Ordinary Least Squares (OLS) Multiple Linear Regression.

    NOTE: Because Purchase_Intention is binary (0/1), this is technically
    a Linear Probability Model (LPM). Coefficients represent the
    change in PROBABILITY of intending to purchase for a 1-unit
    rise in each predictor. A supplementary Logistic Regression
    is printed below for robustness.

    Interpretation guide
    ─────────────────────
    R²          : Proportion of variance in DV explained by the model.
                  R² = 0.25 → model explains 25 % of variance.
    Adj. R²     : R² penalised for the number of predictors — prefer this
                  for comparing models of different sizes.
    F-statistic : Tests whether the overall model fits better than an
                  intercept-only model. Significant p < 0.05 → model is useful.
    Beta (coef) : Unstandardised regression coefficient. A 1-unit increase
                  in IV leads to β-unit change in DV, holding others constant.
    p-value     : Probability of observing the coefficient by chance if the
                  null hypothesis (β = 0) is true.
                  p < 0.05 → statistically significant at the 5 % level.
    """
    print(f"\n{'='*70}")
    print("STEP 8 ── MULTIPLE LINEAR REGRESSION (OLS / Linear Probability Model)")
    print(f"{'='*70}")
    print(f"  DV  : {dv}")
    print(f"  IVs : {ivs}\n")

    subset = df[[dv] + ivs].dropna()
    X = sm.add_constant(subset[ivs])
    y = subset[dv]

    model  = sm.OLS(y, X).fit()
    print(model.summary())

    # ── Standardised Beta coefficients ───────────────────────────────────────
    print("\n  Standardised Beta Coefficients (for effect-size comparison):")
    print(f"  {'Variable':<30} {'Std Beta':>10}")
    print(f"  {'-'*42}")
    for iv in ivs:
        std_beta = model.params[iv] * (subset[iv].std() / subset[dv].std())
        print(f"  {iv:<30} {std_beta:>10.4f}")

    return model


def run_logistic(df: pd.DataFrame, dv: str, ivs: list):
    """
    Supplementary Binary Logistic Regression.
    Reports Odds Ratios (OR) and 95 % confidence intervals.

    Interpretation:
      OR > 1  → predictor increases the odds of Purchase_Intention = 1
      OR < 1  → predictor decreases the odds
      OR = 1  → no effect
    """
    print(f"\n{'='*70}")
    print("SUPPLEMENTARY ── BINARY LOGISTIC REGRESSION")
    print("(Appropriate given binary DV; use alongside MLR results)")
    print(f"{'='*70}")

    subset = df[[dv] + ivs].dropna()
    X = sm.add_constant(subset[ivs])
    y = subset[dv]

    logit  = sm.Logit(y, X).fit(disp=0)
    print(logit.summary())

    # Odds Ratios + 95 % CI
    OR = np.exp(logit.params)
    CI = np.exp(logit.conf_int())
    CI.columns = ["OR_Lower_95CI", "OR_Upper_95CI"]
    or_table = pd.concat([OR.rename("Odds_Ratio"), CI], axis=1)
    print("\n  Odds Ratios and 95 % Confidence Intervals:")
    print(or_table.round(4).to_string())

    return logit


# ── Assumption Checks ─────────────────────────────────────────────────────────

def check_vif(df: pd.DataFrame, ivs: list) -> pd.DataFrame:
    """
    Variance Inflation Factor (VIF) — Multicollinearity check.

    Interpretation:
      VIF < 5   → acceptable (no concerning multicollinearity)
      VIF 5–10  → moderate multicollinearity — investigate further
      VIF > 10  → severe multicollinearity — model may be unreliable
    """
    print(f"\n{'='*70}")
    print("STEP 10a ── MULTICOLLINEARITY CHECK (VIF)")
    print(f"{'='*70}")
    print("  Threshold: VIF < 5 (ideal) | VIF > 10 (problematic)")

    subset = df[ivs].dropna()
    X      = sm.add_constant(subset)
    vif_data = pd.DataFrame({
        "Variable": subset.columns,
        "VIF": [variance_inflation_factor(X.values, i + 1)
                for i in range(len(subset.columns))]
    })
    vif_data["Verdict"] = vif_data["VIF"].apply(
        lambda v: "OK" if v < 5 else ("Moderate" if v < 10 else "HIGH — Problematic")
    )
    print(vif_data.round(4).to_string(index=False))
    return vif_data


def check_normality(residuals: np.ndarray, save_path: str):
    """
    Normality of residuals — OLS assumption.
    Tests used:
      • Shapiro-Wilk (W) — preferred for n < 2000
      • Kolmogorov-Smirnov (D)
    Visual: Q-Q plot + residual histogram.

    Interpretation:
      p > 0.05 → fail to reject H₀ (residuals are approximately normal) ✓
      p < 0.05 → residuals deviate from normality (check histogram/Q-Q shape)
    """
    print(f"\n{'='*70}")
    print("STEP 10b ── NORMALITY OF RESIDUALS")
    print(f"{'='*70}")

    sw_stat, sw_p = stats.shapiro(residuals)
    ks_stat, ks_p = stats.kstest(residuals, "norm",
                                  args=(np.mean(residuals), np.std(residuals)))

    print(f"  Shapiro-Wilk  : W = {sw_stat:.4f}, p = {sw_p:.4f}  "
          f"→ {'Normal ✓' if sw_p > 0.05 else 'Non-normal ✗'}")
    print(f"  Kolmogorov-Smirnov: D = {ks_stat:.4f}, p = {ks_p:.4f}  "
          f"→ {'Normal ✓' if ks_p > 0.05 else 'Non-normal ✗'}")
    print("  Note: With a binary DV, residuals will not be perfectly normal —")
    print("        this is expected; the logistic regression is the primary model.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[0].scatter(osm, osr, color="#2196F3", alpha=0.6, s=20, label="Residuals")
    axes[0].plot(osm, slope * np.array(osm) + intercept,
                 "r--", linewidth=1.5, label="Reference line")
    axes[0].set_title("Normal Q-Q Plot of Residuals", fontweight="bold")
    axes[0].set_xlabel("Theoretical Quantiles")
    axes[0].set_ylabel("Sample Quantiles")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Histogram
    axes[1].hist(residuals, bins=20, color="#4CAF50", edgecolor="white",
                 alpha=0.8, density=True)
    x = np.linspace(residuals.min(), residuals.max(), 200)
    axes[1].plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)),
                 "r-", linewidth=2, label="Normal curve")
    axes[1].set_title("Histogram of Residuals", fontweight="bold")
    axes[1].set_xlabel("Residual value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Normality of Residuals Diagnostics",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Normality plots saved → {save_path}")


def check_homoscedasticity(fitted: np.ndarray,
                            residuals: np.ndarray,
                            save_path: str):
    """
    Homoscedasticity — residuals should have constant variance across fitted values.
    Breusch-Pagan test + Residuals vs Fitted scatter plot.

    Interpretation:
      BP p > 0.05 → homoscedasticity assumption holds ✓
      BP p < 0.05 → heteroscedasticity present (use robust standard errors)
    """
    print(f"\n{'='*70}")
    print("STEP 10c ── HOMOSCEDASTICITY CHECK")
    print(f"{'='*70}")

    from statsmodels.stats.diagnostic import het_breuschpagan
    X_bp = sm.add_constant(fitted)
    bp_stat, bp_p, f_stat, f_p = het_breuschpagan(residuals, X_bp)

    print(f"  Breusch-Pagan: LM = {bp_stat:.4f}, p = {bp_p:.4f}  "
          f"→ {'Homoscedastic ✓' if bp_p > 0.05 else 'Heteroscedastic ✗ (use robust SE)'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, residuals, color="#9C27B0", alpha=0.55, s=25, edgecolors="white")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Fitted Values", fontsize=11)
    ax.set_ylabel("Residuals", fontsize=11)
    ax.set_title("Residuals vs Fitted Values\n(Homoscedasticity Check)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Homoscedasticity plot saved → {save_path}")


def check_autocorrelation(residuals: np.ndarray):
    """
    Durbin-Watson test for autocorrelation in residuals.

    Interpretation:
      DW ≈ 2.0  → no autocorrelation ✓
      DW < 1.5  → positive autocorrelation ✗
      DW > 2.5  → negative autocorrelation ✗
    """
    dw = durbin_watson(residuals)
    print(f"\n  Durbin-Watson statistic: {dw:.4f}  "
          f"→ {'No autocorrelation ✓' if 1.5 <= dw <= 2.5 else 'Autocorrelation present ✗'}")


# ── Save Outputs ──────────────────────────────────────────────────────────────

def save_text_report(model, logit, corr_df: pd.DataFrame,
                     reliability_df: pd.DataFrame,
                     desc_df: pd.DataFrame,
                     vif_df: pd.DataFrame,
                     save_path: str):
    """Save regression summaries, correlation matrix, and reliability table to .txt."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("GREEN BRANDING & GEN Z PURCHASE BEHAVIOR — ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("── CRONBACH'S ALPHA ──\n")
        f.write(reliability_df.to_string(index=False))
        f.write("\n\n")

        f.write("── DESCRIPTIVE STATISTICS ──\n")
        f.write(desc_df.to_string())
        f.write("\n\n")

        f.write("── PEARSON CORRELATION MATRIX ──\n")
        f.write(corr_df.round(3).to_string())
        f.write("\n\n")

        f.write("── VIF TABLE ──\n")
        f.write(vif_df.round(4).to_string(index=False))
        f.write("\n\n")

        f.write("── OLS REGRESSION SUMMARY ──\n")
        f.write(str(model.summary()))
        f.write("\n\n")

        f.write("── LOGISTIC REGRESSION SUMMARY ──\n")
        f.write(str(logit.summary()))
        f.write("\n")

    print(f"\n  Full text report saved → {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df_raw = load_data(DATASET_PATH)

    # ── 2. Clean — Age filter ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 3 ── DATA CLEANING")
    print(f"{'='*70}")
    df = clean_age(df_raw)

    # ── 3. Convert Likert + encode DV ─────────────────────────────────────────
    df = convert_likert(df, LIKERT_COLS, REVERSE_ITEMS)
    df = encode_dv(df)

    # Columns needed for analysis
    all_items      = list(LIKERT_COLS.keys())
    construct_vars = list(CONSTRUCTS.keys())
    analysis_cols  = all_items + ["Purchase_Intention"]

    df = drop_missing(df, analysis_cols)

    # Verify encoding
    print(f"\n  Purchase Intention value counts:\n{df['Purchase_Intention'].value_counts().to_string()}")
    print(f"\n  Likert item sample (first 5 rows):")
    print(df[all_items[:6]].head().to_string())

    # ── 4. Create composite construct scores ──────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 4 ── COMPOSITE CONSTRUCT VARIABLES (row-wise item means)")
    print(f"{'='*70}")
    df = make_composites(df, CONSTRUCTS)
    print(f"  Constructs created: {construct_vars}")
    print(df[construct_vars + ['Purchase_Intention']].describe().round(3).to_string())

    # ── 5. Cronbach's Alpha ────────────────────────────────────────────────────
    reliability_df = reliability_report(df, CONSTRUCTS)

    # ── 6. Descriptive Statistics ──────────────────────────────────────────────
    desc_df = descriptive_stats(df, construct_vars)

    # ── 7. Correlation Matrix + Heatmap ───────────────────────────────────────
    corr_df = correlation_heatmap(
        df, construct_vars,
        save_path=os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    )

    # ── 8 & 9. Multiple Linear Regression ─────────────────────────────────────
    IVS = construct_vars   # all four constructs as predictors
    ols_model = run_mlr(df, dv="Purchase_Intention", ivs=IVS)

    # ── Supplementary: Logistic Regression ────────────────────────────────────
    logit_model = run_logistic(df, dv="Purchase_Intention", ivs=IVS)

    # ── 10. Assumption Checks ─────────────────────────────────────────────────
    residuals = ols_model.resid.values
    fitted    = ols_model.fittedvalues.values

    vif_df = check_vif(df, IVS)

    check_normality(
        residuals,
        save_path=os.path.join(OUTPUT_DIR, "normality_diagnostics.png")
    )

    check_homoscedasticity(
        fitted, residuals,
        save_path=os.path.join(OUTPUT_DIR, "homoscedasticity_plot.png")
    )

    check_autocorrelation(residuals)

    # ── 11. Export cleaned dataset ─────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
    save_cols = (
        [AGE_COL, "Gender"]
        + all_items
        + construct_vars
        + ["Purchase_Intention"]
    )
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"STEP 11 ── CLEANED DATASET EXPORTED → {csv_path}")
    print(f"{'='*70}")

    # ── 12. Save text report ──────────────────────────────────────────────────
    save_text_report(
        ols_model, logit_model,
        corr_df, reliability_df, desc_df, vif_df,
        save_path=os.path.join(OUTPUT_DIR, "analysis_report.txt")
    )

    print(f"\n{'='*70}")
    print("✅  PIPELINE COMPLETE — all outputs written to ./{OUTPUT_DIR}/")
    print(f"{'='*70}")
    print("  Files generated:")
    for fname in os.listdir(OUTPUT_DIR):
        print(f"    • {os.path.join(OUTPUT_DIR, fname)}")


# =============================================================================
if __name__ == "__main__":
    main()