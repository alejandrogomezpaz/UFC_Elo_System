"""Vercel serverless entrypoint: a thin Flask wrapper around the existing
query pipeline in `4. Model Query Pipeline/predict.py` (imported untouched).

Routes
    GET /               predictor page
    GET /methodology    how the model works
    GET /authors        who built it
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

# --------------------------------------------------------------------------
# Vercel path handoff.
#
# vercel.json rewrites every URL to this function (`/(.*)` -> `/api/index`).
# Depending on how the platform hands the request to a WSGI app, PATH_INFO can
# arrive as the *rewritten* path ("/api/index"), or empty (""), instead of the
# URL the visitor actually asked for. Either one makes every Flask route miss
# and Werkzeug serves its default "Not Found" page for the whole site.
#
# Vercel forwards the original request URL in x-vercel-original-path / the
# standard x-forwarded-* set, so prefer that, then fall back to repairing the
# two degenerate PATH_INFO values above.
# --------------------------------------------------------------------------

_FUNCTION_PATH = "/api/index"


class _RestoreRequestPath:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO") or ""

        # Only intervene when the path is unusable as-is; a correct handoff is
        # left completely untouched.
        if path in ("", _FUNCTION_PATH) or path.startswith(_FUNCTION_PATH + "/"):
            original = (environ.get("HTTP_X_VERCEL_ORIGINAL_PATH")
                        or environ.get("HTTP_X_ORIGINAL_URI")
                        or "")
            if original:
                original = original.split("?", 1)[0]

            if original and not original.startswith(_FUNCTION_PATH):
                path = original
            elif path.startswith(_FUNCTION_PATH + "/"):
                # "/api/index/rankings" -> "/rankings"
                path = path[len(_FUNCTION_PATH):]
            else:
                path = "/"

            environ["PATH_INFO"] = path if path.startswith("/") else "/" + path

        # SCRIPT_NAME must stay empty, or url_for() prefixes every generated link.
        environ["SCRIPT_NAME"] = ""
        return self.wsgi_app(environ, start_response)


app.wsgi_app = _RestoreRequestPath(app.wsgi_app)

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


# human-readable labels for the 34 model features (keys match feature_cols)
_PRETTY = {
    "rating_diff": "Glicko-2 rating", "rating_deviation_diff": "rating uncertainty (RD)",
    "volatility_diff": "rating volatility",
    "sig_str_pct_norm_diff": "sig. strike accuracy", "sig_str_landed_norm_diff": "sig. strikes landed",
    "sig_str_attempted_norm_diff": "sig. strikes attempted",
    "total_str_landed_norm_diff": "total strikes landed",
    "total_str_attempted_norm_diff": "total strikes attempted",
    "head_landed_norm_diff": "head strikes landed", "head_attempted_norm_diff": "head strikes attempted",
    "body_landed_norm_diff": "body strikes landed", "body_attempted_norm_diff": "body strikes attempted",
    "leg_landed_norm_diff": "leg strikes landed", "leg_attempted_norm_diff": "leg strikes attempted",
    "distance_landed_norm_diff": "distance strikes landed",
    "distance_attempted_norm_diff": "distance strikes attempted",
    "clinch_landed_norm_diff": "clinch strikes landed",
    "clinch_attempted_norm_diff": "clinch strikes attempted",
    "td_pct_norm_diff": "takedown accuracy", "td_landed_norm_diff": "takedowns landed",
    "td_attempted_norm_diff": "takedowns attempted", "sub_att_norm_diff": "submission attempts",
    "rev_norm_diff": "reversals", "ground_landed_norm_diff": "ground strikes landed",
    "ground_attempted_norm_diff": "ground strikes attempted", "ctrl_secs_norm_diff": "control time",
    "wins_diff": "career wins", "losses_diff": "career losses",
    "reach_z_diff": "reach (class z-score)", "height_inches_z_diff": "height (class z-score)",
    "age_diff": "age",
    "stance_Orthodox_diff": "orthodox stance", "stance_Southpaw_diff": "southpaw stance",
    "stance_Switch_diff": "switch stance",
}


def _pretty(col: str) -> str:
    return _PRETTY.get(col, col.replace("_diff", "").replace("_norm", "").replace("_", " "))


@app.get("/api/analytics")
def api_analytics():
    import numpy as np
    coefs = predict.model.coef_[0]
    col_theme = {c: t for t, cols in predict.themes.items() for c in cols}
    # coefficient spread across the 1,000 bootstrap refits
    ens = np.array([m.coef_[0] for _, m in predict.ensemble])
    lo, hi = np.percentile(ens, [2.5, 97.5], axis=0)
    sel = (ens != 0).mean(axis=0)  # how often the lasso keeps each feature

    feats = [{
        "name": col, "label": _pretty(col),
        "theme": col_theme.get(col, "other"),
        "coef": float(coefs[i]),
        "lo": float(lo[i]), "hi": float(hi[i]),
        "sel": float(sel[i]),
    } for i, col in enumerate(predict.feature_cols)]
    feats.sort(key=lambda f: -abs(f["coef"]))

    return jsonify(
        intercept=float(predict.model.intercept_[0]),
        n_features=len(feats),
        n_zero=int((coefs == 0).sum()),
        n_bootstrap=len(predict.ensemble),
        features=feats,
    )


# --------------------------------------------------------------------------
# Rankings: Glicko-2 leaderboards read straight from fighter_features.
# Snapshots are PRE-fight, so a fighter's newest row is the rating they carried
# INTO their most recent bout -- stated plainly on the page.
# --------------------------------------------------------------------------

_ACTIVE_MONTHS = 24          # "current" = fought within this window
_DIV_LIMITS = [(115, "Strawweight"), (125, "Flyweight"), (135, "Bantamweight"),
               (145, "Featherweight"), (155, "Lightweight"), (170, "Welterweight"),
               (185, "Middleweight"), (205, "Light Heavyweight")]
_DIV_ORDER = [d for _, d in _DIV_LIMITS] + ["Heavyweight"]


def _division(weight):
    # the db stores contracted weight in lbs, catchweights included; snap each
    # to the nearest official limit, everything above light heavy is heavyweight
    if weight is None:
        return "Unknown"
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return "Unknown"
    if w > 206:
        return "Heavyweight"
    return min(_DIV_LIMITS, key=lambda lim: abs(w - lim[0]))[1]


# peak = the single highest-rated snapshot a fighter ever held (ties -> earliest)
_SQL_PEAK = """
SELECT DISTINCT ON (fighter) fighter, rating, rating_deviation, date, weight
FROM fighter_features
WHERE rating IS NOT NULL
ORDER BY fighter, rating DESC, date ASC
"""

# current = each fighter's newest snapshot, restricted to recent activity
_SQL_CURRENT = """
SELECT DISTINCT ON (fighter) fighter, rating, rating_deviation, date, weight
FROM fighter_features
WHERE rating IS NOT NULL
  AND date >= (CURRENT_DATE - make_interval(months => :months))
ORDER BY fighter, date DESC
"""


@app.get("/api/rankings")
def api_rankings():
    try:
        with _get_engine().connect() as conn:
            peak = conn.execute(text(_SQL_PEAK)).mappings().all()
            cur = conn.execute(text(_SQL_CURRENT),
                               {"months": _ACTIVE_MONTHS}).mappings().all()
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    except Exception as e:
        return jsonify(error=str(e)[:300]), 503

    def shape(rs, limit):
        out = [{
            "fighter": r["fighter"],
            "rating": round(float(r["rating"]), 1),
            "rd": None if r["rating_deviation"] is None else round(float(r["rating_deviation"]), 1),
            "date": str(r["date"]),
            "division": _division(r["weight"]),
        } for r in rs]
        out.sort(key=lambda d: -d["rating"])
        return out[:limit]

    return jsonify(divisions=_DIV_ORDER, window_months=_ACTIVE_MONTHS,
                   alltime=shape(peak, 800), current=shape(cur, 1200))


# --------------------------------------------------------------------------
# Frontend: single-file pages sharing one base template. Tokens (__TITLE__,
# __NAV__, __CONTENT__) are substituted with str.replace, so CSS/JS braces
# need no escaping.
# --------------------------------------------------------------------------

_BASE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0' stop-color='%235b8cff'/%3E%3Cstop offset='1' stop-color='%23b44bff'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='32' cy='32' r='28' fill='url(%23g)'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#06070f; --ink:#e9eaf6; --muted:#9a9db8; --faint:#6b6e8a;
    --line:rgba(255,255,255,.09); --glass:rgba(255,255,255,.045);
    --blue:#5b8cff; --violet:#8b5bff; --pink:#c77dff;
    --grad:linear-gradient(100deg,#5b8cff,#8b5bff 55%,#c77dff);
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    font-family:'Inter',system-ui,sans-serif; color:var(--ink); background:var(--bg);
    min-height:100vh; line-height:1.6; -webkit-font-smoothing:antialiased;
  }
  h1,h2,h3,.brand{font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-.01em}
  a{color:var(--blue); text-decoration:none}
  a:hover{text-decoration:underline}

  /* ---- ethereal animated background ---- */
  .bg{position:fixed; inset:0; z-index:-2; overflow:hidden; background:
      radial-gradient(120% 90% at 50% 0%, #0b0d22 0%, #06070f 60%)}
  .blob{position:absolute; border-radius:50%; filter:blur(80px); opacity:.5;
        mix-blend-mode:screen; will-change:transform}
  .b1{width:58vmax; height:58vmax; top:-18vmax; left:-12vmax;
      background:radial-gradient(circle at 35% 35%, #2b4bff, transparent 62%);
      animation:drift1 26s ease-in-out infinite alternate}
  .b2{width:52vmax; height:52vmax; bottom:-20vmax; right:-10vmax;
      background:radial-gradient(circle at 65% 45%, #6d2bff, transparent 62%);
      animation:drift2 34s ease-in-out infinite alternate}
  .b3{width:38vmax; height:38vmax; top:28%; left:56%; opacity:.32;
      background:radial-gradient(circle at 50% 50%, #b44bff, transparent 65%);
      animation:drift3 40s ease-in-out infinite alternate}
  .b4{width:30vmax; height:30vmax; top:55%; left:-8vmax; opacity:.25;
      background:radial-gradient(circle at 50% 50%, #3b6dff, transparent 65%);
      animation:drift2 30s ease-in-out infinite alternate-reverse}
  @keyframes drift1{to{transform:translate(9vmax,7vmax) scale(1.14)}}
  @keyframes drift2{to{transform:translate(-8vmax,-9vmax) scale(1.10)}}
  @keyframes drift3{to{transform:translate(-10vmax,6vmax) scale(.9)}}
  .bg::after{content:""; position:absolute; inset:0;
    background-image:radial-gradient(rgba(255,255,255,.06) 1px, transparent 1px);
    background-size:34px 34px;
    -webkit-mask-image:radial-gradient(ellipse 90% 60% at 50% 12%, #000 30%, transparent 80%);
            mask-image:radial-gradient(ellipse 90% 60% at 50% 12%, #000 30%, transparent 80%)}
  @media (prefers-reduced-motion:reduce){.blob{animation:none}}

  /* ---- nav ---- */
  nav{position:sticky; top:0; z-index:10; display:flex; align-items:center; gap:1rem;
      padding:.8rem clamp(1rem,4vw,2.2rem);
      background:rgba(6,7,15,.55); backdrop-filter:blur(14px);
      -webkit-backdrop-filter:blur(14px); border-bottom:1px solid var(--line)}
  .brand{display:flex; align-items:center; gap:.55rem; font-weight:700; font-size:1.05rem;
         color:var(--ink)}
  .brand:hover{text-decoration:none}
  .dot{width:.9rem; height:.9rem; border-radius:50%; background:var(--grad);
       box-shadow:0 0 14px rgba(139,91,255,.8)}
  .navlinks{margin-left:auto; display:flex; gap:.25rem}
  .navlinks a{color:var(--muted); font-size:.92rem; font-weight:500;
              padding:.42rem .85rem; border-radius:999px; transition:all .18s}
  .navlinks a:hover{color:var(--ink); background:rgba(255,255,255,.06); text-decoration:none}
  .navlinks a.active{color:var(--ink); background:rgba(255,255,255,.10);
                     border:1px solid var(--line)}

  /* ---- layout / cards ---- */
  main{max-width:860px; margin:0 auto; padding:clamp(2rem,6vh,4rem) 1.2rem 4rem}
  .card{background:var(--glass); border:1px solid var(--line); border-radius:18px;
        padding:clamp(1.2rem,3.5vw,2rem); backdrop-filter:blur(14px);
        -webkit-backdrop-filter:blur(14px);
        box-shadow:0 18px 50px rgba(0,0,0,.35); margin-bottom:1.4rem;
        animation:rise .5s ease both}
  @keyframes rise{from{opacity:0; transform:translateY(10px)}to{opacity:1}}
  @media (prefers-reduced-motion:reduce){.card{animation:none}}
  .hero h1{font-size:clamp(1.9rem,5.5vw,3rem); margin:.2rem 0 .6rem; line-height:1.12}
  .grad-text{background:var(--grad); -webkit-background-clip:text; background-clip:text;
             color:transparent}
  .sub{color:var(--muted); max-width:56ch; margin:0}
  .eyebrow{display:inline-block; font-size:.75rem; font-weight:600; letter-spacing:.14em;
           text-transform:uppercase; color:var(--pink); margin-bottom:.4rem}
  h2{font-size:1.25rem; margin:0 0 .5rem}
  h3{font-size:1.02rem; margin:1.1rem 0 .3rem}
  p{margin:.5rem 0}
  ul{margin:.5rem 0 0; padding-left:1.2rem}
  li{margin:.4rem 0}
  li::marker{color:var(--violet)}
  .muted{color:var(--muted)}
  code,.math{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em;
       background:rgba(255,255,255,.07); border:1px solid var(--line);
       padding:.1em .4em; border-radius:6px; white-space:nowrap}
  .stats{display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.2rem}
  .chip{font-size:.8rem; color:var(--muted); border:1px solid var(--line);
        background:rgba(255,255,255,.03); padding:.35rem .8rem; border-radius:999px}
  footer{text-align:center; color:var(--faint); font-size:.8rem; padding:0 1rem 2rem}

  /* ---- predictor form ---- */
  .fightform{display:grid; grid-template-columns:1fr auto 1fr; gap:.7rem; align-items:center;
             margin-top:1.4rem}
  .vs{font-family:'Space Grotesk'; font-weight:700; color:var(--faint); text-align:center}
  input{width:100%; padding:.8rem 1rem; font-size:1rem; font-family:inherit; color:var(--ink);
        background:rgba(255,255,255,.05); border:1px solid var(--line); border-radius:12px;
        outline:none; transition:border-color .18s, box-shadow .18s}
  input::placeholder{color:var(--faint)}
  input:focus{border-color:var(--violet); box-shadow:0 0 0 3px rgba(139,91,255,.22)}
  .predictbtn{grid-column:1/-1; justify-self:center; margin-top:.4rem;
        padding:.8rem 2.6rem; font-size:1rem; font-weight:600; font-family:inherit;
        color:#fff; background:var(--grad); border:none; border-radius:999px; cursor:pointer;
        box-shadow:0 8px 26px rgba(109,63,255,.4); transition:transform .15s, box-shadow .15s}
  .predictbtn:hover{transform:translateY(-1px); box-shadow:0 12px 32px rgba(109,63,255,.55)}
  .predictbtn:disabled{opacity:.6; cursor:wait; transform:none}
  .examples{margin-top:1.1rem; font-size:.83rem; color:var(--faint)}
  .examples button{font:inherit; color:var(--muted); background:none; cursor:pointer;
        border:1px solid var(--line); border-radius:999px; padding:.25rem .7rem;
        margin:.2rem .15rem; transition:all .15s}
  .examples button:hover{color:var(--ink); border-color:var(--violet)}
  @media (max-width:560px){.fightform{grid-template-columns:1fr}.vs{display:none}}

  /* ---- results ---- */
  .headline{font-family:'Space Grotesk'; font-size:clamp(1.4rem,4.5vw,2rem); font-weight:700;
            margin:0 0 .2rem}
  .snapshots{color:var(--faint); font-size:.82rem; margin:0 0 1.1rem}
  .bar{position:relative; height:14px; border-radius:999px; overflow:hidden;
       background:rgba(255,255,255,.08); margin:.5rem 0 .3rem}
  .bar .fill{position:absolute; inset:0 auto 0 0; background:var(--grad);
       border-radius:999px 0 0 999px; transition:width .8s cubic-bezier(.2,.8,.2,1)}
  .bar .ci{position:absolute; top:0; bottom:0; background:rgba(255,255,255,.25);
       border-left:1px solid rgba(255,255,255,.55); border-right:1px solid rgba(255,255,255,.55)}
  .barlabels{display:flex; justify-content:space-between; font-size:.88rem; color:var(--muted)}
  .barlabels b{color:var(--ink); font-weight:600}
  .ci-note{color:var(--faint); font-size:.78rem; margin:.2rem 0 0}
  pre.reason{background:rgba(0,0,0,.32); border:1px solid var(--line); border-radius:12px;
       padding:1rem 1.1rem; white-space:pre-wrap; font-size:.86rem; line-height:1.55;
       color:#c6c9e2; margin:1.1rem 0 0}
  .err{color:#ff8fa3; background:rgba(255,79,116,.08); border:1px solid rgba(255,79,116,.3);
       border-radius:12px; padding:.8rem 1rem; margin-top:1rem}
  .loading{display:flex; gap:.45rem; align-items:center; color:var(--muted); margin-top:1.2rem}
  .loading span{width:.5rem; height:.5rem; border-radius:50%; background:var(--violet);
       animation:pulse 1s ease-in-out infinite}
  .loading span:nth-child(2){animation-delay:.15s}
  .loading span:nth-child(3){animation-delay:.3s}
  @keyframes pulse{0%,100%{opacity:.25; transform:scale(.8)}50%{opacity:1; transform:scale(1)}}

  /* ---- analytics ---- */
  .legend{display:flex; flex-wrap:wrap; gap:.4rem 1rem; margin:.5rem 0 1rem; font-size:.78rem;
          color:var(--muted)}
  .sw{display:inline-block; width:.65rem; height:.65rem; border-radius:3px; margin-right:.35rem;
      vertical-align:-1px}
  .axis{display:grid; grid-template-columns:minmax(110px,175px) 1fr 52px; gap:.6rem;
        font-size:.72rem; color:var(--faint); margin-bottom:.35rem}
  .axis .mid{display:flex; justify-content:space-between}
  .frow{display:grid; grid-template-columns:minmax(110px,175px) 1fr 52px; gap:.6rem;
        align-items:center; margin:.3rem 0; font-size:.82rem}
  .fname{color:var(--muted); text-align:right; white-space:nowrap; overflow:hidden;
         text-overflow:ellipsis}
  .ftrack{position:relative; height:12px}
  .ftrack::before{content:""; position:absolute; left:50%; top:-3px; bottom:-3px; width:1px;
         background:rgba(255,255,255,.18)}
  .fbar{position:absolute; top:1px; height:10px; border-radius:5px; opacity:.92}
  .fci{position:absolute; top:5px; height:2px; background:rgba(255,255,255,.4); border-radius:1px}
  .fval{color:var(--faint); font-variant-numeric:tabular-nums; font-size:.76rem}
  .trow{display:grid; grid-template-columns:minmax(110px,175px) 1fr 52px; gap:.6rem;
        align-items:center; margin:.45rem 0}
  .ttrack{height:12px; border-radius:6px; background:rgba(255,255,255,.06); overflow:hidden}
  .tfill{height:100%; border-radius:6px}
  .zeroed{margin-top:1rem; font-size:.8rem; color:var(--faint); line-height:1.9}
  .zeroed .chip{margin-right:.25rem}

  /* ---- authors ---- */
  .author{display:flex; gap:1.3rem; align-items:flex-start; flex-wrap:wrap}
  .avatar{width:84px; height:84px; border-radius:50%; background:var(--grad); flex-shrink:0;
       display:flex; align-items:center; justify-content:center; font-family:'Space Grotesk';
       font-weight:700; font-size:1.7rem; color:#fff;
       box-shadow:0 0 30px rgba(139,91,255,.45)}
  .author .info{flex:1; min-width:240px}
  .role{color:var(--pink); font-size:.85rem; font-weight:600; margin:.1rem 0 .6rem}
</style></head><body>
<div class="bg"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="blob b4"></div></div>
<nav>
  <a class="brand" href="/"><span class="dot"></span>UFC Fight Predictor</a>
  <div class="navlinks">__NAV__</div>
</nav>
<main>
__CONTENT__
</main>
<footer>Built on 10,900+ UFC fights &middot; predictions are statistical estimates, not betting advice.</footer>
</body></html>"""

_NAV_ITEMS = [("Predict", "/"), ("Rankings", "/rankings"), ("Analytics", "/analytics"),
              ("Methodology", "/methodology"), ("Author", "/author")]


def _render(title: str, content: str, active: str) -> str:
    nav = "".join(
        f'<a href="{href}"{cls}>{label}</a>'
        for label, href in _NAV_ITEMS
        for cls in [' class="active"' if label == active else ""]
    )
    return (_BASE.replace("__TITLE__", title)
                 .replace("__NAV__", nav)
                 .replace("__CONTENT__", content))


_PREDICT_CONTENT = """
<div class="card hero">
  <h1>Counterfactual Prediction Model</h1>
  <form id="f" class="fightform">
    <input id="a" placeholder="Fighter A" required autocomplete="off">
    <div class="vs">VS</div>
    <input id="b" placeholder="Fighter B" required autocomplete="off">
    <button class="predictbtn" id="go">Predict</button>
  </form>
  <div class="examples">Try:
    <button type="button" data-a="Khabib Nurmagomedov" data-b="Conor McGregor">Khabib vs McGregor</button>
    <button type="button" data-a="Demetrious Johnson" data-b="Henry Cejudo">Johnson vs Cejudo</button>
    <button type="button" data-a="Israel Adesanya" data-b="Alex Pereira">Adesanya vs Pereira</button>
  </div>
</div>
<div id="out"></div>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/</g, '&lt;');

document.querySelectorAll('.examples button').forEach(btn => {
  btn.addEventListener('click', () => {
    $('a').value = btn.dataset.a; $('b').value = btn.dataset.b;
    $('f').requestSubmit();
  });
});

function cleanReasoning(t) {
  const lines = String(t || '').replace(/\\r/g, '').split('\\n');
  let i = 0;
  while (i < lines.length && !lines[i].trim()) i++;
  if (i < lines.length) i++;              // drop duplicate headline
  return lines.slice(i).join('\\n').trim();
}

$('f').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const out = $('out'), go = $('go');
  go.disabled = true;
  out.innerHTML = '<div class="card loading"><span></span><span></span><span></span>&nbsp;scoring the matchup&hellip;</div>';
  const qs = new URLSearchParams({ fighter_a: $('a').value, fighter_b: $('b').value });
  try {
    const r = await fetch('/api/predict?' + qs);
    const d = await r.json();
    if (!r.ok) {
      out.innerHTML = '<div class="card"><div class="err">' + esc(d.error || 'request failed') + '</div></div>';
      return;
    }
    const pA = d.p_a_wins, lo = d.ci95[0], hi = d.ci95[1];
    out.innerHTML =
      '<div class="card">' +
      '<p class="headline"><span class="grad-text">' + esc(d.favorite) + '</span> wins ' +
        (d.p_favorite * 100).toFixed(1) + '%</p>' +
      '<p class="snapshots">' + esc(d.fighter_a) + ' vs ' + esc(d.fighter_b) +
        ' &mdash; pre-fight snapshots ' + esc(d.snapshot_a) + ' / ' + esc(d.snapshot_b) + '</p>' +
      '<div class="bar"><div class="ci" style="left:' + (lo * 100).toFixed(1) + '%;width:' +
        ((hi - lo) * 100).toFixed(1) + '%"></div>' +
        '<div class="fill" style="width:0%"></div></div>' +
      '<div class="barlabels"><span><b>' + esc(d.fighter_a) + '</b> ' + (pA * 100).toFixed(1) +
        '%</span><span>' + ((1 - pA) * 100).toFixed(1) + '% <b>' + esc(d.fighter_b) + '</b></span></div>' +
      '<p class="ci-note">shaded band: 95% bootstrap CI for ' + esc(d.fighter_a) + ' [' +
        (lo * 100).toFixed(1) + '%&ndash;' + (hi * 100).toFixed(1) + '%]</p>' +
      '<pre class="reason">' + esc(cleanReasoning(d.reasoning)) + '</pre>' +
      '</div>';
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        const fill = out.querySelector('.fill');
        if (fill) fill.style.width = (pA * 100).toFixed(1) + '%';
      }));
    out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    out.innerHTML = '<div class="card"><div class="err">network error</div></div>';
  } finally {
    go.disabled = false;
  }
});
</script>
"""

_ANALYTICS_CONTENT = """
<div class="card hero">
  <span class="eyebrow">What the model has learned</span>
  <h1>Analytics</h1>
  <p class="sub">The lasso's surviving coefficients, read straight off the live model. All
  features are standardized, so bar lengths are directly comparable: each shows how hard one
  standard deviation of advantage in that stat pulls the predicted log-odds.</p>
</div>
<div id="an"><div class="card loading"><span></span><span></span><span></span>&nbsp;loading coefficients&hellip;</div></div>
<script>
const THEME_COLORS = {
  'glicko rating': '#5b8cff', 'striking': '#8b5bff', 'grappling': '#c77dff',
  'record': '#4fd8c9', 'physical/age': '#ff8fb8', 'stance': '#7f86ff', 'other': '#9a9db8'
};
const esc = (s) => String(s).replace(/</g, '&lt;');
const f2 = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);

(async () => {
  const el = document.getElementById('an');
  let d;
  try {
    const r = await fetch('/api/analytics');
    d = await r.json();
    if (!r.ok) throw new Error(d.error || 'request failed');
  } catch (e) {
    el.innerHTML = '<div class="card"><div class="err">' + esc(e.message) + '</div></div>';
    return;
  }
  const active = d.features.filter(f => f.coef !== 0);
  const zeroed = d.features.filter(f => f.coef === 0);
  const max = Math.max(...active.map(f => Math.max(Math.abs(f.coef), Math.abs(f.lo), Math.abs(f.hi))));
  const pos = (v) => 50 + (v / max) * 48;          // percent position of value v

  // theme totals (sum of |coef|)
  const totals = {};
  active.forEach(f => { totals[f.theme] = (totals[f.theme] || 0) + Math.abs(f.coef); });
  const tmax = Math.max(...Object.values(totals));
  const themeRows = Object.entries(totals).sort((a, b) => b[1] - a[1]).map(([t, v]) =>
    '<div class="trow"><div class="fname">' + esc(t) + '</div>' +
    '<div class="ttrack"><div class="tfill" style="width:' + (v / tmax * 100).toFixed(1) +
    '%;background:' + (THEME_COLORS[t] || '#9a9db8') + '"></div></div>' +
    '<div class="fval">' + v.toFixed(2) + '</div></div>').join('');

  const legend = Object.keys(totals).map(t =>
    '<span><span class="sw" style="background:' + (THEME_COLORS[t] || '#9a9db8') +
    '"></span>' + esc(t) + '</span>').join('');

  const rows = active.map(f => {
    const c = THEME_COLORS[f.theme] || '#9a9db8';
    const L = pos(Math.min(0, f.coef)), R = pos(Math.max(0, f.coef));
    const cl = pos(Math.min(f.lo, f.hi)), cr = pos(Math.max(f.lo, f.hi));
    const tip = esc(f.label) + ': β = ' + f2(f.coef) + ' | 95% bootstrap range ' +
                f2(f.lo) + ' to ' + f2(f.hi) + ' | kept in ' + Math.round(f.sel * 100) +
                '% of 1,000 refits';
    return '<div class="frow" title="' + tip + '"><div class="fname">' + esc(f.label) + '</div>' +
      '<div class="ftrack">' +
      '<div class="fbar" style="left:' + L.toFixed(2) + '%;width:' + (R - L).toFixed(2) +
        '%;background:' + c + '"></div>' +
      '<div class="fci" style="left:' + cl.toFixed(2) + '%;width:' + Math.max(cr - cl, .4).toFixed(2) +
        '%"></div></div>' +
      '<div class="fval">' + f2(f.coef) + '</div></div>';
  }).join('');

  el.innerHTML =
    '<div class="card"><h2>Where the signal lives</h2>' +
    '<p class="muted">Total influence per theme &mdash; sum of coefficient magnitudes,' +
    ' Σ|β|.</p>' + themeRows + '</div>' +

    '<div class="card"><h2>The biggest factors</h2>' +
    '<p class="muted">' + active.length + ' of ' + d.n_features +
    ' features survived the L1 penalty. Bars: coefficient β on the standardized' +
    ' differential; whiskers: 95% range across ' + d.n_bootstrap.toLocaleString() +
    ' bootstrap refits. Hover any row for details.</p>' +
    '<div class="legend">' + legend + '</div>' +
    '<div class="axis"><div></div><div class="mid"><span>&larr; advantage lowers win odds</span>' +
    '<span>advantage raises win odds &rarr;</span></div><div></div></div>' + rows +
    (zeroed.length ?
      '<div class="zeroed"><b>' + zeroed.length + ' features zeroed out by the lasso:</b> ' +
      zeroed.map(f => '<span class="chip">' + esc(f.label) + '</span>').join('') + '</div>' : '') +
    '</div>' +

    '<div class="card"><h2>How to read this</h2>' +
    '<p>A bar to the right means holding an edge in that stat raises a fighter\\'s predicted' +
    ' win odds; to the left means it lowers them (e.g. more career losses, higher age). Because' +
    ' inputs are standardized, a bar twice as long carries twice the log-odds weight per standard' +
    ' deviation of advantage.</p>' +
    '<p class="muted">Whiskers that cross zero mean the bootstrap isn\\'t sure the effect' +
    ' survives resampling &mdash; treat those as weak signals. The kept-rate in each tooltip' +
    ' shows how often the lasso selected the feature across refits: a rough measure of' +
    ' stability. Coefficients update automatically whenever the model retrains.</p></div>';
})();
</script>
"""

_RANKINGS_CONTENT = """
<style>
  main{max-width:1020px}
  .tabs{display:flex; gap:.4rem; margin:1.3rem 0 .3rem; flex-wrap:wrap}
  .tab{font:inherit; font-size:.92rem; font-weight:600; color:var(--muted); cursor:pointer;
       background:rgba(255,255,255,.04); border:1px solid var(--line); border-radius:999px;
       padding:.45rem 1.1rem; transition:all .18s}
  .tab:hover{color:var(--ink)}
  .tab.on{color:#fff; background:var(--grad); border-color:transparent;
          box-shadow:0 6px 20px rgba(109,63,255,.35)}
  .chips{display:flex; gap:.35rem; flex-wrap:wrap; margin:.9rem 0 .2rem}
  .chipbtn{font:inherit; font-size:.8rem; color:var(--muted); cursor:pointer;
       background:rgba(255,255,255,.03); border:1px solid var(--line); border-radius:999px;
       padding:.32rem .8rem; transition:all .15s}
  .chipbtn:hover{color:var(--ink); border-color:var(--violet)}
  .chipbtn.on{color:var(--ink); background:rgba(139,91,255,.22); border-color:var(--violet)}
  .searchbox{margin-top:.9rem}
  .searchbox input{padding:.6rem .9rem; font-size:.9rem}
  table.rank{width:100%; border-collapse:collapse; margin-top:1rem; font-size:.9rem}
  table.rank th{text-align:left; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase;
       color:var(--faint); font-weight:600; padding:.5rem .6rem; border-bottom:1px solid var(--line)}
  table.rank td{padding:.55rem .6rem; border-bottom:1px solid rgba(255,255,255,.05)}
  table.rank tr:hover td{background:rgba(255,255,255,.035)}
  .num{text-align:right; font-variant-numeric:tabular-nums}
  .rk{color:var(--faint); font-variant-numeric:tabular-nums; width:2.6rem}
  .rk.top{color:var(--pink); font-weight:700}
  .who{font-weight:600}
  .rating{font-family:'Space Grotesk'; font-weight:600}
  .div{color:var(--muted); font-size:.84rem}
  .when{color:var(--faint); font-size:.84rem; font-variant-numeric:tabular-nums}
  .rdcell{color:var(--faint); font-variant-numeric:tabular-nums}
  .empty{color:var(--faint); padding:1.2rem .2rem}
  @media (max-width:620px){
    table.rank .hide-sm{display:none}
    table.rank td,table.rank th{padding:.5rem .35rem}
  }
</style>

<div class="card hero">
  <span class="eyebrow">Glicko-2 leaderboards</span>
  <h1>Rankings</h1>
  <p class="sub">Ratings are computed on one global scale across every division, so they double
  as a pound-for-pound list. Pick a tab, then filter by division.</p>
  <div class="tabs">
    <button class="tab on" id="t-current">Current</button>
    <button class="tab" id="t-alltime">All-time peaks</button>
  </div>
  <div class="chips" id="chips"></div>
  <div class="searchbox"><input id="q" placeholder="Search a fighter&hellip;" autocomplete="off"></div>
</div>
<div id="rk"><div class="card loading"><span></span><span></span><span></span>&nbsp;loading ratings&hellip;</div></div>
<script>
const esc = (s) => String(s).replace(/</g, '&lt;');
const state = { tab: 'current', div: 'P4P', q: '' };
let DATA = null;

const depth = () => (state.div === 'P4P' ? 25 : 15);

function rows() {
  const src = DATA[state.tab] || [];
  const pool = state.div === 'P4P' ? src : src.filter(r => r.division === state.div);
  const ranked = pool.slice(0, depth()).map((r, i) => ({ r, rank: i + 1 }));
  if (!state.q) return ranked;
  const q = state.q.toLowerCase();
  // search looks through the whole (filtered) pool, keeping true ranks visible
  return pool.map((r, i) => ({ r, rank: i + 1 }))
             .filter(o => o.r.fighter.toLowerCase().includes(q))
             .slice(0, 50);
}

function table() {
  const list = rows();
  if (!list.length) return '<div class="card"><p class="empty">Nobody matches that filter.</p></div>';
  const peak = state.tab === 'alltime';
  const head =
    '<tr><th class="rk">#</th><th>Fighter</th>' +
    '<th class="num">' + (peak ? 'Peak rating' : 'Rating') + '</th>' +
    '<th class="num hide-sm">RD</th>' +
    '<th class="hide-sm">' + (peak ? 'Peak date' : 'Last fight') + '</th>' +
    '<th>Weight class</th></tr>';
  const body = list.map(o =>
    '<tr><td class="rk' + (o.rank <= 3 ? ' top' : '') + '">' + o.rank + '</td>' +
    '<td class="who">' + esc(o.r.fighter) + '</td>' +
    '<td class="num rating">' + o.r.rating.toFixed(1) + '</td>' +
    '<td class="num rdcell hide-sm">' + (o.r.rd == null ? '&mdash;' : '&plusmn;' + o.r.rd.toFixed(0)) + '</td>' +
    '<td class="when hide-sm">' + esc(o.r.date) + '</td>' +
    '<td class="div">' + esc(o.r.division) + '</td></tr>').join('');
  const note = peak
    ? 'Highest rating each fighter ever carried, with the date they held it and the division they were fighting in at the time.'
    : 'Latest rating for fighters who have competed in the last ' + DATA.window_months +
      ' months. Snapshots are pre-fight, so this is the rating a fighter took into their most recent bout.';
  return '<div class="card">' +
    '<h2>' + (peak ? 'All-time peak ratings' : 'Current ratings') +
    ' &mdash; ' + esc(state.div === 'P4P' ? 'pound-for-pound' : state.div) + '</h2>' +
    '<p class="muted">' + note + ' RD is the rating deviation: the model&rsquo;s uncertainty about ' +
    'that number, so a big RD means few or long-ago fights &mdash; read those rows with caution.</p>' +
    '<table class="rank"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
}

function chips() {
  const all = ['P4P'].concat(DATA.divisions);
  document.getElementById('chips').innerHTML = all.map(d =>
    '<button class="chipbtn' + (d === state.div ? ' on' : '') + '" data-d="' + esc(d) + '">' +
    esc(d) + '</button>').join('');
  document.querySelectorAll('.chipbtn').forEach(b => b.addEventListener('click', () => {
    state.div = b.dataset.d; chips(); draw();
  }));
}

function draw() { document.getElementById('rk').innerHTML = table(); }

function setTab(tab) {
  state.tab = tab;
  document.getElementById('t-current').classList.toggle('on', tab === 'current');
  document.getElementById('t-alltime').classList.toggle('on', tab === 'alltime');
  draw();
}

document.getElementById('t-current').addEventListener('click', () => setTab('current'));
document.getElementById('t-alltime').addEventListener('click', () => setTab('alltime'));
document.getElementById('q').addEventListener('input', (e) => {
  state.q = e.target.value.trim(); draw();
});

(async () => {
  try {
    const r = await fetch('/api/rankings');
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'request failed');
    DATA = d;
  } catch (e) {
    document.getElementById('rk').innerHTML =
      '<div class="card"><div class="err">' + esc(e.message) + '</div></div>';
    return;
  }
  chips();
  draw();
})();
</script>
"""


_METHODOLOGY_CONTENT = """
<div class="card hero">
  <span class="eyebrow">Under the hood</span>
  <h1>Methodology</h1>
  <p class="sub">Every design choice serves two goals: no information leakage, and predictions
  a human can interrogate. Each section below gives the intuition first, then the notation.</p>
</div>

<div class="card">
  <h2>1 &middot; Data</h2>
  <p>The dataset is built by a tree search of the UFC Stats website (Playwright + BeautifulSoup),
  then deduplicated and normalized &mdash; 10,900+ fights with full round-by-round statistics.
  Crucially, every feature attached to a fight uses only information available <em>before</em>
  that fight: each fighter is represented by a pre-fight snapshot, so the model never peeks at
  the outcome it is trying to predict.</p>
  <p class="muted">Snapshots live in a Neon-hosted PostgreSQL database, and an updater re-scrapes
  new events, recomputes ratings, and refreshes the model automatically.</p>
</div>

<div class="card">
  <h2>2 &middot; Skill ratings: Glicko-2</h2>
  <p>The Glicko-2 rating system quantifies skill level, uncertainty, and volatility. A win over an
  uncertain opponent moves you less than a win over a well-established one, and long layoffs inflate
  uncertainty &mdash; both natural fits for MMA's sparse fight schedules. This is a cool usage of
  distributions!</p>
  <p>Formally, each fighter carries a rating <code>&mu;</code>, a rating deviation
  <code>&phi;</code> (standard error of &mu;), and a volatility <code>&sigma;</code>. The system
  constant <code>&tau;</code>, which governs how fast volatility can change, was hyperparameter-tuned
  by minimizing out-of-sample log loss rather than left at a textbook default.</p>
</div>

<div class="card">
  <h2>3 &middot; Feature engineering</h2>
  <p>Beyond ratings, each snapshot captures how a fighter has actually been performing: striking
  and grappling output, defense, record, and physical attributes. Recent fights should matter more
  than old ones, so running averages are time-weighted. Physical stats are normalized within weight
  class &mdash; being tall <em>for a flyweight</em> is what matters, which also makes the ratings
  scale comparable across divisions.</p>
  <p>A matchup is then encoded as a differential: the model sees
  <code>x = f_A &minus; f_B</code>, a 34-dimensional vector of feature differences. This bakes in
  a useful symmetry &mdash; swapping the two fighters exactly flips the prediction.</p>
</div>

<div class="card">
  <h2>4 &middot; Model</h2>
  <p>Given the highly correlated nature of the pre-engineered data (e.g. total strikes aggregated
  from body strikes + leg strikes + &hellip;) I used the Elastic-Net logistic regression model which
  penalizes large weights (Ridge) and zaps low value coefficients to zero (Lasso). This is a highly
  interpretable model. Interestingly the model converged to a purely Lasso logistic regression model
  when trained through log-loss minimization.</p>
  <p>The win probability is <code>p(A beats B) = &sigma;(&beta;&#8320; + &beta;&#7511;x)</code>
  where <code>&sigma;</code> is the logistic function, fit by minimizing
  <code>&Sigma; log-loss + &lambda;&#8214;&beta;&#8214;&#8321;</code> over 10,900+ fights, with
  <code>&lambda;</code> chosen by cross-validation.</p>
</div>

<div class="card">
  <h2>5 &middot; Uncertainty: bootstrap ensemble</h2>
  <p>A single 63% is misleading if the model would say anything from 51% to 75% under slightly
  different training data. To measure that fragility, the training set is resampled with
  replacement 1,000 times and a model is refit on each resample. The spread of those 1,000
  predictions is the confidence interval shown with every pick.</p>
  <p>The reported band is the percentile interval: the 2.5th and 97.5th percentiles of
  <code>{p&#8321;, &hellip;, p&#8321;&#8320;&#8320;&#8320;}</code>, a 95% bootstrap CI.</p>
</div>

<div class="card">
  <h2>6 &middot; Natural language reasoning</h2>
  <p>Every prediction ships with a plain-English explanation generated directly from the Lasso
  coefficients &mdash; no post-hoc approximation. Because the model is linear in log-odds, its
  reasoning decomposes exactly: each feature contributes <code>&beta;&#11388; &middot; x&#11388;</code>
  (on scaled differentials), positive terms pull toward fighter A, negative toward B, and they
  simply add up. Contributions are grouped into themes &mdash; rating, striking, grappling,
  record, physical, stance &mdash; and the explanation is deterministic: same input, same words.</p>
</div>

<div class="card">
  <h2>7 &middot; Serving</h2>
  <p>The live system has four moving parts:</p>
  <ul>
    <li><b>Database</b> &mdash; Neon-hosted PostgreSQL storing every fighter's pre-fight
    feature snapshots.</li>
    <li><b>Query pipeline</b> &mdash; <code>predict.py</code>: name two fighters, it pulls their
    latest snapshots, forms the differential vector, and scores it through the model and
    bootstrap ensemble.</li>
    <li><b>Vercel app</b> &mdash; a Flask serverless function wrapping that pipeline; this site
    and its JSON API (<code>/api/predict</code>).</li>
    <li><b>Updater</b> &mdash; <code>updater.py</code>: re-scrapes new events, recomputes
    Glicko-2 ratings, appends to the database, and refreshes the model.</li>
  </ul>
</div>
"""

_AUTHORS_CONTENT = """
<div class="card">
  <div class="author">
    <div class="avatar">AG</div>
    <div class="info">
      <h2>Alejandro (Alex) Gomez-Paz</h2>
      <p class="role">University of Washington &mdash; B.S. Applied Mathematics, Data Science option
      &middot; Expected June 2028</p>
      <p class="muted">Seattle, Washington</p>
      <p class="muted">
        <a href="mailto:alexgp@uw.edu">alexgp@uw.edu</a> &middot;
        <a href="https://github.com/alejandrogomezpaz" target="_blank" rel="noopener">GitHub</a> &middot;
        <a href="https://www.linkedin.com/in/alejandro-gomez-paz/" target="_blank" rel="noopener">LinkedIn</a>
      </p>
    </div>
  </div>
  <p>An end-to-end, automatically updating UFC fight-outcome prediction model built on scraped,
  leakage-free data, pairing a bootstrapped logistic regression with natural-language reasoning
  &mdash; written in Python and SQL.</p>
</div>
"""


@app.get("/")
def home():
    return _render("UFC Fight Predictor", _PREDICT_CONTENT, "Predict")


@app.get("/analytics")
def analytics():
    return _render("Analytics — UFC Fight Predictor", _ANALYTICS_CONTENT, "Analytics")


@app.get("/rankings")
def rankings():
    return _render("Rankings — UFC Fight Predictor", _RANKINGS_CONTENT, "Rankings")


@app.get("/methodology")
def methodology():
    return _render("Methodology — UFC Fight Predictor", _METHODOLOGY_CONTENT, "Methodology")


@app.get("/author")
@app.get("/authors")          # legacy path, kept so old links don't 404
def author():
    return _render("Author — UFC Fight Predictor", _AUTHORS_CONTENT, "Author")


@app.errorhandler(404)
def not_found(_e):
    # A branded 404 instead of Werkzeug's default. It echoes the path Flask
    # actually received, which makes a Vercel path-handoff problem obvious at a
    # glance rather than looking like the whole site is down.
    content = f"""
<div class="card hero">
  <h1>Page not found</h1>
  <p class="sub">No route matches <code>{request.path}</code>.</p>
  <div class="examples">Go to:
    <button type="button" onclick="location.href='/'">Predict</button>
    <button type="button" onclick="location.href='/rankings'">Rankings</button>
    <button type="button" onclick="location.href='/analytics'">Analytics</button>
  </div>
</div>
"""
    return _render("Not found — UFC Fight Predictor", content, ""), 404


@app.get("/api/whoami")
def whoami():
    # Diagnostic: shows exactly what the WSGI layer handed Flask.
    return jsonify(
        path=request.path,
        raw_path_info=request.environ.get("PATH_INFO"),
        script_name=request.environ.get("SCRIPT_NAME"),
        vercel_original_path=request.headers.get("x-vercel-original-path"),
        host=request.host,
        routes=sorted(str(r.rule) for r in app.url_map.iter_rules()),
    )
