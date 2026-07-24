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

_NAV_ITEMS = [("Predict", "/"), ("Methodology", "/methodology"), ("Authors", "/authors")]


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
  <span class="eyebrow">Counterfactual matchup engine</span>
  <h1>Who wins? <span class="grad-text">Ask the model.</span></h1>
  <p class="sub">Glicko-2 ratings + L1 logistic regression trained on 10,900+ UFC fights,
  with a 1,000-model bootstrap confidence interval. Predictions are generated live from
  Neon Postgres.</p>
  <form id="f" class="fightform">
    <input id="a" placeholder="Fighter A (e.g. Jon Jones)" required autocomplete="off">
    <div class="vs">VS</div>
    <input id="b" placeholder="Fighter B (e.g. Stipe Miocic)" required autocomplete="off">
    <button class="predictbtn" id="go">Predict</button>
  </form>
  <div class="examples">Try:
    <button type="button" data-a="Jon Jones" data-b="Stipe Miocic">Jones vs Miocic</button>
    <button type="button" data-a="Islam Makhachev" data-b="Alexander Volkanovski">Makhachev vs Volkanovski</button>
    <button type="button" data-a="Alex Pereira" data-b="Israel Adesanya">Pereira vs Adesanya</button>
  </div>
  <div class="stats">
    <span class="chip">10,900+ fights</span>
    <span class="chip">34 engineered features</span>
    <span class="chip">1,000-model bootstrap CI</span>
    <span class="chip">Deterministic explanations</span>
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
  <p>Elo answers &ldquo;how good is this fighter?&rdquo; with a single number. Glicko-2 adds two
  more: how <em>sure</em> we are about that number, and how <em>volatile</em> the fighter's
  performances are. A win over an uncertain opponent moves you less than a win over a
  well-established one, and long layoffs inflate uncertainty &mdash; both natural fits for MMA's
  sparse fight schedules.</p>
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
  <h2>4 &middot; Model: L1-regularized logistic regression</h2>
  <p>A deliberately simple model. Logistic regression keeps every prediction interpretable, and the
  L1 (lasso) penalty forces the model to be opinionated: features that don't pull their weight get
  coefficients of exactly zero, leaving a sparse, readable set of drivers.</p>
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
  <h2>6 &middot; Explanations, not vibes</h2>
  <p>Because the model is linear in log-odds, its reasoning decomposes exactly &mdash; no
  post-hoc approximation. Each feature contributes <code>&beta;&#11388; &middot; x&#11388;</code>
  (on scaled differentials) to the final log-odds; positive terms pull toward fighter A, negative
  toward B, and they simply add up. Contributions are grouped into themes &mdash; rating, striking,
  grappling, record, physical, stance &mdash; and the same input always produces the same
  explanation, word for word.</p>
</div>

<div class="card">
  <h2>7 &middot; Serving</h2>
  <p>Name two fighters and the pipeline pulls their latest pre-fight snapshots from Postgres,
  forms the differential vector, scores it through the model and the bootstrap ensemble, and
  returns probability, CI, and reasoning &mdash; live, in one request, on a Vercel serverless
  function.</p>
</div>
"""

_AUTHORS_CONTENT = """
<div class="card hero">
  <span class="eyebrow">The team</span>
  <h1>Authors</h1>
</div>

<div class="card">
  <div class="author">
    <div class="avatar">AG</div>
    <div class="info">
      <h2>Alejandro (Alex) Gomez-Paz</h2>
      <p class="role">Data science, modeling &amp; engineering &middot; University of Washington</p>
      <p>Alex built this project end to end: scraping and cleaning 10,900+ UFC fights, engineering
      leak-free pre-fight features, tuning the Glicko-2 rating system, training the L1 logistic
      regression and its 1,000-model bootstrap ensemble, and deploying the live prediction service
      you're using now.</p>
      <p>The goal was a fight predictor that is quantitative, objective, and scalable across
      weight classes &mdash; and that can always show its work.</p>
      <p class="muted">Get in touch: <a href="mailto:alexgp@uw.edu">alexgp@uw.edu</a></p>
    </div>
  </div>
</div>
"""


@app.get("/")
def home():
    return _render("UFC Fight Predictor", _PREDICT_CONTENT, "Predict")


@app.get("/methodology")
def methodology():
    return _render("Methodology — UFC Fight Predictor", _METHODOLOGY_CONTENT, "Methodology")


@app.get("/authors")
def authors():
    return _render("Authors — UFC Fight Predictor", _AUTHORS_CONTENT, "Authors")
