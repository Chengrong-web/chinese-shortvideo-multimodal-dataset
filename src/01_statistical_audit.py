import numpy as np
import pandas as pd
from scipy.stats import chisquare

def benford_first_digit_test(series: pd.Series) -> dict:
    clean = series[series > 0].dropna().astype(str)
    first_digits = clean.str.lstrip('0.').str[0].astype(int)
    first_digits = first_digits[first_digits.isin(range(1, 10))]
    observed_counts = first_digits.value_counts().reindex(range(1, 10), fill_value=0).values
    expected_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected_counts = expected_probs * len(first_digits)
    chi2_stat, p_val = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    return {
        "n_records": len(first_digits),
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 4)
    }

def run_audit(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with N={len(df)} records.")
    print("Benford's Law First-Digit Audit:", benford_first_digit_test(df['raw_likes']))
    max_z = df['z_score_outlier_check'].max()
    print(f"Max Z-score: {max_z:.2f}")

if __name__ == "__main__":
    run_audit("data/guochao_multimodal_audited_dataset_N4782.csv")
