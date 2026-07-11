
Logistic Classification Model based on ratings and stylistic term

Rating system chosen: Glicko-2 because unlike Elo, Glicko-2 carries a rating deviation (uncertainty) and a volatility term alongside the rating itself. The rating is necessary because of dynamic opponent strength

Why add a stylistic term. Ratings capture how good a fighter is but not who they match up well against — the sport's well-known "styles make fights" effect (e.g., a grappler neutralizing a superior striker). The stylistic term is learned from the data as a rating-normalized, rating-uncorrelated grouping (striker, grappler, etc.), so by construction it adds signal the rating alone cannot. Because it's orthogonal to rating, it contributes incremental predictive power on top of an already strong rating-only baseline rather than restating it.

Justification of the combination. Rating supplies the dominant main effect (overall skill differential); the stylistic term supplies an interaction the rating structurally can't express. Together they cover both axes of fight prediction — absolute skill and relative matchup — while logistic regression keeps the model interpretable, with each coefficient's sign and magnitude directly readable.