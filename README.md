# UFC_Elo_System
# Hosted on a public website link here: https://ufc-elo-system-en5j.vercel.app

Counter Factual Classification model with:

1) confidence quanitfication via bootstrapping ensamble
2) deterministic natural language reasoning (leveraging Lasso-coefficients)
3) Neon-hosted PostgreSQL database
4) Automatic model and database updater


Pipeline Description:
1) Scraping fight and fighter data by tree search of UFC Stats website (Playwright and Beautiful Soup Python libraries)
2) Data cleaning (mainly deduping, formatting, datatypes, etc.)
3) Feature Engineering
     hyperparameter-tuned rating (Glicko-2) features — volatility constant τ optimized by minimizing log loss
     time-weighted running stat averages, weight-class-normalized biostats (strictly pre-fight, no leakage)
     fighter feature differentials (34-dim A−B vector)
     hosted on Neon-PostgreSQL database
4) L1-regularized logistic regression trained on 10,900+ fights
5) Confidence quantification: 1,000-model bootstrap ensemble → 95% CI on every prediction
6) Natural language reasoning: model is linear in log-odds, so each feature's pull is exactly
   coefficient × scaled differential; contributions grouped into themes (rating, striking,
   grappling, record, physical, stance) — deterministic, same input → same explanation
7) Query pipeline (predict.py): name two fighters → pulls latest pre-fight snapshots from
   Neon, returns win probability + CI + reasoning
8) Updater (updater.py): re-scrapes new events, recomputes Glicko-2 ratings, appends to
   the database, and refreshes the model
