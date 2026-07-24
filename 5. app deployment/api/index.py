"""Vercel serverless entrypoint: a thin Flask wrapper around the existing
query pipeline in `4. Model Query Pipeline/predict.py` (imported untouched).

Routes
    GET /               tiny HTML page to query the model from a browser
    GET /api/health     app + database reachability check
    GET /api/predict    ?fighter_a=..&fighter_b=..  -> JSON prediction

Database: reads os.environ["DATABASE_URL"] (on Vercel, set this to the Neon
*pooled* connection string). Falls back to UFC_DB_URL so existing local
workflows keep working. Credentials are never hardcoded here.
"""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "4. Model Query Pipeline"))
import predict  # noqa: E402  (loads model.joblib once per cold start)

app = Flask(__name__)

_engine = None


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL") or os.environ.get("UFC_DB_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. On Vercel: Project Settings -> "
            "Environment Variables -> DATABASE_URL = Neon pooled connection string."
        )
    # Some dashboards emit the legacy 'postgres://' scheme, which SQLAlchemy 2.x rejects
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def _get_engine():
    global _engine
    if _engine is None:
        # NullPool: Neon's PgBouncer (the -pooler host) already pools; a client-side
        # pool inside a short-lived serverless instance would just hold connections.
        _engine = create_engine(_db_url(), poolclass=NullPool)
    return _engine


@app.get("/api/health")
def health():
    try:
        with _get_engine().connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM fighter_features")).scalar()
    except Exception as e:
        return jsonify(status="db_error", error=str(e)[:300]), 503
    return jsonify(status="ok", fighter_feature_rows=int(n),
                   model_features=len(predict.feature_cols))


@app.get("/api/predict")
def api_predict():
    name_a = (request.args.get("fighter_a") or request.args.get("a") or "").strip()
    name_b = (request.args.get("fighter_b") or request.args.get("b") or "").strip()
    if not name_a or not name_b:
        return jsonify(error="pass ?fighter_a=<name>&fighter_b=<name>"), 400
    if name_a.lower() == name_b.lower():
        return jsonify(error="pick two different fighters"), 400

    try:
        engine = _get_engine()
        # predict.py signals unknown fighters via sys.exit(<hint message>)
        try:
            fA = predict.latest_snapshot(engine, name_a)
            fB = predict.latest_snapshot(engine, name_b)
        except SystemExit as e:
            return jsonify(error=str(e)), 404
    except RuntimeError as e:
        return jsonify(error=str(e)), 503

    for name, snap in ((name_a, fA), (name_b, fB)):
        gone = predict.missing_features(snap)
        if gone:
            return jsonify(error=f"can't score this one: {name} is missing "
                                 f"pre-fight history for {gone}"), 422

    x = predict.diff_vector(fA, fB)
    p, lo, hi = predict.predict(x)

    buf = io.StringIO()
    with redirect_stdout(buf):  # reasoning() prints; capture its output verbatim
        predict.reasoning(x, fA["fighter"], fB["fighter"], p, lo, hi)

    return jsonify(
        fighter_a=fA["fighter"], fighter_b=fB["fighter"],
        snapshot_a=str(fA["date"]), snapshot_b=str(fB["date"]),
        p_a_wins=round(float(p), 4),
        ci95=[round(float(lo), 4), round(float(hi), 4)],
        favorite=fA["fighter"] if p >= 0.5 else fB["fighter"],
        p_favorite=round(float(max(p, 1.0 - p)), 4),
        reasoning=buf.getvalue(),
    )


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UFC Fight Predictor</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:720px;margin:3rem auto;padding:0 1rem;color:#1a1a1a}
  input{padding:.5rem;font-size:1rem;width:14rem;max-width:44vw}
  button{padding:.5rem 1.2rem;font-size:1rem;cursor:pointer}
  pre{background:#f4f4f4;padding:1rem;white-space:pre-wrap;border-radius:6px}
  .err{color:#b00020}
</style></head><body>
<h1>UFC Fight Predictor</h1>
<p>Glicko-2 + L1 logistic regression over 10,900+ fights, with a 1,000-model
bootstrap CI. Data lives in Neon Postgres; predictions are generated live.</p>
<form id="f">
  <input id="a" placeholder="Fighter A" required>
  vs
  <input id="b" placeholder="Fighter B" required>
  <button>Predict</button>
</form>
<div id="out"></div>
<script>
document.getElementById('f').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const out = document.getElementById('out');
  out.innerHTML = '<p>scoring&hellip;</p>';
  const qs = new URLSearchParams({fighter_a: document.getElementById('a').value,
                                  fighter_b: document.getElementById('b').value});
  try {
    const r = await fetch('/api/predict?' + qs);
    const d = await r.json();
    out.innerHTML = r.ok
      ? `<h2>${d.favorite} wins ${(d.p_favorite*100).toFixed(1)}%</h2>
         <p>${d.fighter_a} vs ${d.fighter_b} &mdash; snapshots ${d.snapshot_a} / ${d.snapshot_b}</p>
         <pre>${d.reasoning.replace(/</g,'&lt;')}</pre>`
      : `<p class="err">${(d.error||'request failed').replace(/</g,'&lt;')}</p>`;
  } catch (e) { out.innerHTML = '<p class="err">network error</p>'; }
});
</script></body></html>"""


@app.get("/")
def home():
    return _PAGE
