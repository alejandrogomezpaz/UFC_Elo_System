"""Load fighter_database_instantiation.csv into the local `ufc` Postgres database
as table `fighter_features` with PRIMARY KEY (fighter, date).

Rerunnable: drops and rebuilds the table each run, so you can re-execute it
after every rescrape / feature-engineering pass.

Usage:
    pip install pandas sqlalchemy psycopg2-binary   (first time only)
    python load_fighter_features.py
"""

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# Defaults to local Postgres.app; set UFC_DB_URL to load a cloud database instead.
# Keep cloud URLs in the environment, never in this file: they contain a password
# and this folder is a git repo.
DB_URL = os.environ.get(
    "UFC_DB_URL", "postgresql+psycopg2://alejandrogomez-paz@localhost:5432/ufc"
)
TABLE = "fighter_features"
CSV = (
    Path(__file__).resolve().parent.parent
    / "3. feature_engineering_and_model_training"
    / "fighter_database_instantiation.csv"
)


def main() -> None:
    # index_col=0 drops the unnamed pandas index column saved in the CSV
    df = pd.read_csv(CSV, index_col=0)

    # Real date types instead of strings
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce").dt.date

    # Lowercase column names (stance_Orthodox -> stance_orthodox) so you never
    # have to double-quote identifiers in SQL
    df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]

    # Guard: Postgres will reject the PK anyway, but fail early with a clear message
    dupes = df.duplicated(subset=["fighter", "date"]).sum()
    if dupes:
        raise SystemExit(
            f"Aborting: {dupes} duplicate (fighter, date) rows in {CSV.name}. "
            "Fix the CSV, then rerun."
        )

    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        df.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.execute(text(f"ALTER TABLE {TABLE} ADD PRIMARY KEY (fighter, date)"))

    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
    print(f"Loaded {n:,} rows into '{TABLE}' with PRIMARY KEY (fighter, date).")


if __name__ == "__main__":
    main()
