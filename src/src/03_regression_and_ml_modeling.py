import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_models(csv_path: str):
    df = pd.read_csv(csv_path)
    df['is_trending_audio'] = (df['audio_type'] == 'Trending Audio').astype(int)
    df['ln_followers'] = np.log(df['author_followers'] + 1)

    formula_rq1 = (
        "ln_engagement_index ~ cutting_speed_cuts_per_min + ocr_text_coverage_pct + "
        "human_face_present + is_trending_audio + speech_rate_words_per_min + "
        "ln_followers + duration_seconds + C(platform)"
    )
    ols_model = smf.ols(formula=formula_rq1, data=df).fit()
    print(f"Adjusted R2: {ols_model.rsquared_adj:.4f} | F-stat: {ols_model.fvalue:.2f}")

    formula_rq2 = (
        "circulation_velocity_vcirc_pct ~ ocr_text_coverage_pct * has_ecommerce_tag + "
        "cutting_speed_cuts_per_min + is_trending_audio + C(platform)"
    )
    inter_model = smf.ols(formula=formula_rq2, data=df).fit()
    print(inter_model.summary().tables[1])

    features = [
        'is_trending_audio', 'human_face_present', 'cutting_speed_cuts_per_min',
        'danmu_density_msgs_per_sec', 'ln_followers', 'ocr_text_coverage_pct',
        'has_ecommerce_tag', 'duration_seconds'
    ]
    X, y = df[features], df['ln_engagement_index']
    rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv = cross_validate(rf, X, y, cv=kf, scoring=('r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'))
    print(f"RF R2: {cv['test_r2'].mean():.4f} | MAE: {-cv['test_neg_mean_absolute_error'].mean():.4f} | RMSE: {-cv['test_neg_root_mean_squared_error'].mean():.4f}")

    cluster_cols = ['cutting_speed_cuts_per_min', 'ocr_text_coverage_pct', 'is_trending_audio', 'has_ecommerce_tag']
    X_scaled = StandardScaler().fit_transform(df[cluster_cols])
    df['cluster'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled)
    print(df.groupby('cluster')[['ln_engagement_index', 'circulation_velocity_vcirc_pct']].agg(['count', 'mean']))

if __name__ == "__main__":
    run_models("data/guochao_multimodal_audited_dataset_N4782.csv")
