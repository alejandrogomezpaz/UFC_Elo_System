import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

'''
query pipeline: name two fighters, get a win probability with receipts
    1) pull each fighter's latest pre-fight snapshot from postgres (fighter_features)
    2) build the A-minus-B diff vector exactly like feature_engineering_instantiator
    3) score with the saved logistic model + 95% CI from the 1000 bootstrap models
    4) reasoning chunk: logistic regression is linear in log-odds, so every feature's
       pull on the prediction is exactly coef * scaled_diff. the explanation below is
       read straight off the model -- deterministic, same input -> same words every time
convention
    diff = A - B and p = P(A wins), matching training.csv (y=1 means fighter A won)
usage
    python predict.py "Fighter A" "Fighter B"     (set UFC_DB_URL to hit the cloud db)
'''

DB_URL = os.environ.get('UFC_DB_URL', 'postgresql+psycopg2://alejandrogomez-paz@localhost:5432/ufc')

bundle = joblib.load(Path(__file__).resolve().parent / 'model.joblib')
scaler, model, ensemble, feature_cols = bundle['scaler'], bundle['model'], bundle['ensemble'], bundle['feature_cols']

# how the reasoning chunk groups the 34 features into themes a human actually thinks in
themes = {
    'glicko rating': ['rating_diff', 'rating_deviation_diff', 'volatility_diff'],
    'striking':      ['sig_str_pct_norm_diff', 'sig_str_landed_norm_diff', 'sig_str_attempted_norm_diff',
                      'total_str_landed_norm_diff', 'total_str_attempted_norm_diff',
                      'head_landed_norm_diff', 'head_attempted_norm_diff',
                      'body_landed_norm_diff', 'body_attempted_norm_diff',
                      'leg_landed_norm_diff', 'leg_attempted_norm_diff',
                      'distance_landed_norm_diff', 'distance_attempted_norm_diff',
                      'clinch_landed_norm_diff', 'clinch_attempted_norm_diff'],
    'grappling':     ['td_pct_norm_diff', 'td_landed_norm_diff', 'td_attempted_norm_diff',
                      'sub_att_norm_diff', 'rev_norm_diff', 'ground_landed_norm_diff',
                      'ground_attempted_norm_diff', 'ctrl_secs_norm_diff'],
    'record':        ['wins_diff', 'losses_diff'],
    'physical/age':  ['reach_z_diff', 'height_inches_z_diff', 'age_diff'],
    'stance':        ['stance_Orthodox_diff', 'stance_Southpaw_diff', 'stance_Switch_diff'],
}


def latest_snapshot(engine, name):
    q = text('SELECT * FROM fighter_features WHERE fighter = :name ORDER BY date DESC LIMIT 1')
    rows = pd.read_sql(q, engine, params={'name': name})
    if len(rows) == 1:
        return rows.iloc[0]
    close = pd.read_sql(text('SELECT DISTINCT fighter FROM fighter_features WHERE fighter ILIKE :pat ORDER BY fighter'),
                        engine, params={'pat': f'%{name}%'})
    hints = ', '.join(close['fighter'].head(5)) if len(close) else 'no close matches'
    sys.exit(f"'{name}' is not in fighter_features. did you mean: {hints}")


def missing_features(snapshot):
    # model was trained with debut/sparse rows dropped, so it can't score NaNs honestly
    return [c[:-5] for c in feature_cols if pd.isna(snapshot[c[:-5].lower()])]


def diff_vector(fA, fB):
    # db columns are lowercase (stance_orthodox), feature_cols keep the notebook casing
    x = {col: fA[col[:-5].lower()] - fB[col[:-5].lower()] for col in feature_cols}
    return pd.DataFrame([x])


def predict(x):
    p = model.predict_proba(scaler.transform(x))[0, 1]
    ps = np.array([m.predict_proba(sc.transform(x))[0, 1] for sc, m in ensemble])
    return p, np.percentile(ps, 2.5), np.percentile(ps, 97.5)


def reasoning(x, name_a, name_b, p):
    # exact bookkeeping, no vibes: logit(p) = intercept + sum(coef * scaled_diff),
    # so each feature's pull is its coef * scaled_diff term. + pulls toward A, - toward B
    pulls = pd.Series(model.coef_[0] * scaler.transform(x)[0], index=feature_cols)
    favorite = name_a if p >= 0.5 else name_b
    zeroed = int((model.coef_[0] == 0).sum())

    print(f'\nwhy the model picks {favorite}:')
    print(f'    (log-odds pulls; + favors {name_a}, - favors {name_b}. '
          f'lasso zeroed {zeroed}/{len(feature_cols)} features, the survivors do the talking)')

    totals = {t: pulls[cols].sum() for t, cols in themes.items()}
    for theme, pull in sorted(totals.items(), key=lambda kv: (-abs(kv[1]), kv[0])):
        verdict = 'a wash' if abs(pull) < 0.05 else f'favors {name_a if pull > 0 else name_b} ({pull:+.2f})'
        print(f'    {theme:<14} {verdict}')

    top = pulls.reindex(pulls.abs().sort_values(ascending=False, kind='stable').index)[:3]
    print(f'    biggest single factors: ' + ', '.join(f'{c} ({v:+.2f})' for c, v in top.items()))

    logit = model.intercept_[0] + pulls.sum()
    print(f'    net: {logit:+.2f} log-odds -> p = {1 / (1 + np.exp(-logit)):.3f}, hence the call')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: python predict.py "Fighter A" "Fighter B"')
    name_a, name_b = sys.argv[1], sys.argv[2]
    if name_a == name_b:
        sys.exit('pick two different fighters')

    engine = create_engine(DB_URL)
    fA = latest_snapshot(engine, name_a)
    fB = latest_snapshot(engine, name_b)

    for name, snap in [(name_a, fA), (name_b, fB)]:
        gone = missing_features(snap)
        if gone:
            sys.exit(f"can't score this one: {name} is missing pre-fight history for {gone} "
                     f"(debut or sparse data -- the model was trained without these fights)")

    x = diff_vector(fA, fB)
    p, lo, hi = predict(x)
    favorite, p_fav = (name_a, p) if p >= 0.5 else (name_b, 1 - p)

    print(f'\n{name_a} vs {name_b}')
    print(f'    snapshots: {name_a} as of {fA["date"]}, {name_b} as of {fB["date"]}')
    print(f'    P({name_a} wins) = {p:.3f}    95% CI [{lo:.3f}, {hi:.3f}]')
    print(f'    call: {favorite} ({p_fav:.0%})')
    reasoning(x, name_a, name_b, p)
