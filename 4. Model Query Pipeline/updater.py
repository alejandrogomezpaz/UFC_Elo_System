import datetime
import os
import sys
from pathlib import Path

import glicko2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

'''
incremental updater: scrape ufcstats.com for new events, clean, and roll the
glicko ratings + features forward in postgres. run it after every card.
    1) takes a date (or date range) as the scraping window
    2) asks the db for the last date it was updated
    3) scrapes ufcstats.com with playwright, starting from the least recent
       event not yet in the db, iterating forward chronologically
    4) per event: cleans the scraped data (cleaning code copy-pasted from
       2. data_cleaning/data_cleaning.ipynb), then per fight writes the fight
       rows, the round-by-round stats, and both fighters' new pre-fight
       snapshots (glicko + features) into the db
    5) repeats event by event until caught up through the input date
tables it maintains (fights/fight_stats/fighters are created and seeded from
the local cleaned csvs the first time this runs against a fresh db):
    fights           -- fighters_fights.csv schema, two rows per fight
    fight_stats      -- fights.csv schema, round-by-round + round 0 aggregate
    fighters         -- fighters.csv schema, static biostats
    fighter_features -- must already exist (load_fighter_features.py); new
                        pre-fight snapshot rows get appended here
conventions (same as instantiation):
    snapshots are PRE-fight: row (fighter, date) is the state carried INTO the
    fight on that date, so a fight never leaks into its own features. draws
    ('TIE') count for neither wins nor losses; no-contests/overturned fights
    crash the scraper on purpose and are discarded, same policy as always.
failsafes / redundancy:
    - always resumes from the db's own last date; a requested start after that
      is clamped down, since a gap would silently corrupt every rating after it
    - one transaction per event: a crash mid-event rolls back cleanly and the
      next run picks up exactly where it left off
    - fight_id and (fighter, date) existence checks make reruns idempotent
    - unscrapeable fights are logged to failed_updates.csv, never crash the run
    - table schemas are introspected up front; aborts loudly on any mismatch
    - glicko carry replays every fight since the last stored snapshot, so a
      hole in fighter_features self-heals instead of compounding
    - post-run verification cross-checks max dates and row counts across tables
usage
    python updater.py                         catch up through today
    python updater.py 2026-07-23              catch up through a given date
    python updater.py 2026-06-01 2026-07-23   explicit window (start clamps to db state)
    set UFC_DB_URL to hit neon; defaults to the local postgres.app db.
    first time: pip install pandas numpy beautifulsoup4 playwright sqlalchemy
    psycopg2-binary glicko2, then: playwright install chromium
'''

DB_URL = os.environ.get('UFC_DB_URL', 'postgresql+psycopg2://alejandrogomez-paz@localhost:5432/ufc')

# converged in ratings.ipynb; static, do not re-fit here
OPTIMAL_TAU = 0.10040325212871097
glicko2.Player._tau = OPTIMAL_TAU

HERE = Path(__file__).resolve().parent
CLEANING_DIR = HERE.parent / '2. data_cleaning'
FAILED_CSV = str(HERE / 'failed_updates.csv')

# fight-page method strings -> the fighter-page vocabulary stored in fights.
# method_type is kept for completeness; ratings and features never read it
METHOD_MAP = {
    'Decision - Unanimous': 'U-DEC',
    'Decision - Split': 'S-DEC',
    'Decision - Majority': 'M-DEC',
    'KO/TKO': 'KO/TKO',
    "TKO - Doctor's Stoppage": 'KO/TKO',
    'Submission': 'SUB',
    'DQ': 'DQ',
    'Could Not Continue': 'CNC',
    'Overturned': 'Overturned',
}

# the per-minute stat columns, copied from feature_engineering_instantiator.ipynb
NORM_COLS = ['sig_str_pct', 'td_pct', 'sub_att', 'rev', 'sig_str_landed', 'sig_str_attempted',
             'total_str_landed', 'total_str_attempted', 'td_landed', 'td_attempted',
             'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
             'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
             'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted', 'ctrl_secs']

STANCE_COLS = ['stance_orthodox', 'stance_southpaw', 'stance_switch', 'stance_nan']

TABLE_COLS = {}  # table -> ordered column list, filled by ensure_tables



# ---- scraping (copied from 1. data_scraping/CODE_DATA_SCRAPING.ipynb) ----

def id_from_url(url):
    return url.rstrip('/').split('/')[-1]

def save_progress(df, filename):
    df.to_csv( filename, mode="a", header=not os.path.exists(filename), index=False )

def single_raw_html_fetcher(url, page):
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_selector("body .l-page, table, tbody", timeout=15000)
    except:
        pass
    page.wait_for_timeout(200)
    html = page.content()
    return html

def single_fight_scraper(html):

    soup = BeautifulSoup(html, "html.parser")
    fight_soup = soup.find('div', class_ = 'b-fight-details')


    #For winner/loser/draw info:
    ordered_winner_result = [result.text.strip() for result in fight_soup.find_all('i', class_ = 'b-fight-details__person-status')]
    ordered_fighter_result = [result.text.strip() for result in fight_soup.find_all('a', class_ = 'b-link b-fight-details__person-link')]

    for i in range(2):
        if ordered_winner_result[i] == 'D':
            winner = 'TIE'
            loser = 'TIE'
        elif ordered_winner_result[i] == 'W':
            winner = ordered_fighter_result[i]
        else:
            loser = ordered_fighter_result[i]


    #For fight_details
    fight_details_soup = soup.find('p', class_ = 'b-fight-details__text')
    fight_details_columns = [column.text.strip() for column in fight_details_soup.find_all('i', class_ = 'b-fight-details__label')]

    fight_details_data = []

    fight_details_first_soup = fight_details_soup.find('i', class_ = "b-fight-details__text-item_first")
    fight_details_data.append(fight_details_first_soup.find('i', style="font-style: normal").text.strip())
    labels = fight_details_soup.find_all('i', class_ = 'b-fight-details__label')

    for column in labels:
        fight_details_data.append(column.next_sibling.text.strip())

    fight_details_data.pop(1) #remove blank space
    fight_details_columns.pop(4) #remove referee, since irrelevant
    fight_details_data.pop(4) #removes referee since irrelevant
    df_fight_details = pd.DataFrame([fight_details_data], columns = fight_details_columns)


    #round_by_round table
    rbr_soup = fight_soup.find('tr', class_ = "b-fight-details__table-row") # round by round soup
    rbr_columns = [column.text.strip() for column in rbr_soup] #has '' empty strings
    rbr_columns = [column for column in rbr_columns if column != ''] #removes empty strings

    rbr_fight_data_soup = fight_soup.find_all('section', class_ = "b-fight-details__section js-fight-section")[1]
    rbr_fight_datacell_soup = [column.text.strip() for column in rbr_fight_data_soup.find_all('p', class_ = "b-fight-details__table-text")]

    fighter_one_data = rbr_fight_datacell_soup[::2]
    fighter_two_data = rbr_fight_datacell_soup[1::2]

    df_rbr_total = pd.DataFrame(([fighter_one_data, fighter_two_data]), columns = rbr_columns)

    #aggregate fight table
    #round by round table
    #aggregate significant strikes table
    #round by round sig strikes table

    sections = fight_soup.find_all('section', class_="b-fight-details__section js-fight-section")

    index = [1, 2, 4]
    for i in index:
        section = sections[i]

        rbr_soup = section.find('tr', class_="b-fight-details__table-row")
        rbr_columns = [column.get_text(strip=True) for column in rbr_soup.find_all(['th', 'td'])]

        rbr_data = sections[i].find_all('p', class_="b-fight-details__table-text")
        rbr_data = [column.text.strip() for column in rbr_data]

        fighter_one_data = rbr_data[::2]
        fighter_two_data = rbr_data[1::2]

        arr_one = np.array(fighter_one_data)
        arr1 = arr_one.reshape(len(arr_one) // len(rbr_columns), len(rbr_columns))
        df_fighter_one = pd.DataFrame(arr1, columns=rbr_columns)

        arr_two = np.array(fighter_two_data)
        arr2 = arr_two.reshape(len(arr_two) // len(rbr_columns), len(rbr_columns))
        df_fighter_two = pd.DataFrame(arr2, columns=rbr_columns)

        if i == 1:
            df_aggregate_fight_1 = df_fighter_one
            df_aggregate_fight_2 = df_fighter_two
        if i == 2:
            df_rbr_fight_1 = df_fighter_one
            df_rbr_fight_2 = df_fighter_two
        if i == 4:
            df_sig_strikes_rbr_1 = df_fighter_one
            df_sig_strikes_rbr_2 = df_fighter_two


    #SCHEMA for TABLE

    df_rbr_fight_1.columns = ['Fighter','KD','Sig. str.','Sig. str. %','Total str.','Td','Td %','Sub. att','Rev.','Ctrl'] #reset scraping column names, scraped an error
    df_rbr_fight_2.columns = ['Fighter','KD','Sig. str.','Sig. str. %','Total str.','Td','Td %','Sub. att','Rev.','Ctrl'] #reset scraping column names, scraped an error

    df_aggregate_fight_1["Round"] = 0 #initialized round number
    df_aggregate_fight_2["Round"] = 0

    df_rbr_fight_1['Round'] = df_rbr_fight_1.index + 1
    df_rbr_fight_2['Round'] = df_rbr_fight_2.index + 1


    df_rounds_nonstrikes = pd.concat([df_aggregate_fight_1, df_rbr_fight_1, df_aggregate_fight_2, df_rbr_fight_2], axis = 0) #made composite table

    df_rbr_total['Round'] = 0
    df_sig_strikes_rbr_1['Round'] = df_sig_strikes_rbr_1.index + 1
    df_sig_strikes_rbr_2['Round'] = df_sig_strikes_rbr_2.index + 1

    df_rounds_strikes = pd.concat([df_rbr_total, df_sig_strikes_rbr_1, df_sig_strikes_rbr_2], axis = 0)
    df_rbr_fight  = pd.merge(df_rounds_nonstrikes, df_rounds_strikes, on = ['Fighter', 'Round'])


    df_winner = pd.DataFrame({"winner": [winner], 'loser': [loser]})
    df_fight_stats = pd.concat([df_fight_details, df_winner], axis = 1)

    return  df_fight_stats, df_rbr_fight

def single_fighter_scraper(html):
    soup = BeautifulSoup(html, "html.parser")
    fighter_soup = soup.find('section', class_ = 'b-statistics__section_details')

    #For record
    heading_soup = soup.find('h2')
    record_unclean = heading_soup.find('span', class_ = 'b-content__title-record').text.strip()
    record = record_unclean[8:]
    df_record = pd.DataFrame([record], columns = ['record'])

    #For fight table
    fight_columns = [column.text.strip() for column in fighter_soup.find_all('th')]
    fight_rows = fighter_soup.find_all('tr')

    fight_data = []
    fight_ids = []
    for row in fight_rows:
        cells = row.find_all('td')
        clean_cell = [cell.text.strip() for cell in cells]
        if len(clean_cell) < len(fight_columns):
            clean_cell += [None] * (len(fight_columns) - len(clean_cell))

        fight_data.append(clean_cell)

        link = row.get('data-link')
        fight_ids.append(id_from_url(link) if link else None)

    df_fight_table = pd.DataFrame(fight_data, columns = fight_columns)
    df_fight_table.insert(0, 'fight_id', fight_ids)

    #For biostats table
    biostats_table_soup = fighter_soup.find('ul')
    biostats_columns = [column.text.strip() for column in biostats_table_soup.find_all('i', class_ = 'b-list__box-item-title b-list__box-item-title_type_width')]

    biostats_data = []
    for i in range(len(biostats_columns)):
        biostats_data.append(biostats_table_soup.find_all('i', class_ = 'b-list__box-item-title b-list__box-item-title_type_width')[i].next_sibling.strip())
    df_biostats_table = pd.DataFrame([biostats_data], columns = biostats_columns)


    #For career statistics table
    career_stat_soup = fighter_soup.find('div', class_="b-list__info-box-left")
    career_stat_columns = [column.text.strip() for column in career_stat_soup.find_all('i')]

    career_stat_data = []
    for i in range(len(career_stat_columns)):
        career_stat_data.append(career_stat_soup.find_all('i')[i].next_sibling.strip())
    df_career_stat = pd.DataFrame([career_stat_data], columns = career_stat_columns)

    df_fighter_stats = pd.concat([df_record, df_career_stat, df_biostats_table], axis = 1).drop(columns=["Career statistics:"])

    return df_fighter_stats, df_fight_table


# updater-specific scrapers, same style as the originals

def completed_events_scraper(page):
    #the listing shows every event with its date, newest first (plus the next
    #upcoming card at the top, which the date window filters right back out)
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    html = single_raw_html_fetcher(url, page)
    soup = BeautifulSoup(html, "html.parser")

    table_soup = soup.find('table', class_ = "b-statistics__table-events")
    events = []
    for row in table_soup.find_all('tr', class_ = 'b-statistics__table-row'):
        a_tag = row.find('a', class_ = 'b-link b-link_style_black')
        date_span = row.find('span', class_ = 'b-statistics__date')
        if a_tag is None or date_span is None: #header and spacer rows
            continue
        event_date = pd.to_datetime(date_span.text.strip(), errors='coerce')
        events.append({'event_url': a_tag.get('href'),
                       'event_name': a_tag.text.strip(),
                       'date': None if pd.isna(event_date) else event_date.date()})

    events = [e for e in events if e['date'] is not None]
    events.sort(key=lambda e: e['date'])
    return events

def event_page_parser(html):
    #the event page restates its own date (used to double-check the listing)
    #and holds one data-link per fight, in card order
    soup = BeautifulSoup(html, "html.parser")

    event_date = None
    for li in soup.find_all('li', class_ = 'b-list__box-list-item'):
        item_text = li.get_text(' ', strip=True)
        if item_text.lower().startswith('date:'):
            parsed = pd.to_datetime(item_text[5:].strip(), errors='coerce')
            event_date = None if pd.isna(parsed) else parsed.date()

    fight_urls = []
    event_soup = soup.find('tbody', class_ = "b-fight-details__table-body")
    if event_soup:
        for row in event_soup.find_all('tr'):
            link = row.get('data-link')
            if link and link not in fight_urls: # precautionary dedupe, keeps card order
                fight_urls.append(link)

    return event_date, fight_urls

def fight_person_ids(html):
    #fighter profile links on the fight page, name -> fighter_id
    soup = BeautifulSoup(html, "html.parser")
    ids = {}
    for a_tag in soup.find_all('a', class_ = 'b-link b-fight-details__person-link'):
        link = a_tag.get('href')
        if link:
            ids[a_tag.text.strip()] = id_from_url(link)
    return ids



# ---- cleaning (copied from 2. data_cleaning/data_cleaning.ipynb) ----

def clean_fight_oneline(df_fight_oneline):
    df_fight_oneline.columns = df_fight_oneline.columns.str.replace(':', '').str.replace(' ', '_').str.lower() #reformat col names
    df_fight_oneline['time_sec'] = ((pd.to_numeric(df_fight_oneline['time'].str.split(':', expand=True)[0], errors = 'coerce') * 60) +
                           pd.to_numeric(df_fight_oneline['time'].str.split(':', expand=True)[1], errors = 'coerce')).astype('Int64') #clean time col to secs
    df_fight_oneline = df_fight_oneline.drop(columns = 'time')

    split = df_fight_oneline['time_format'].str.extract(r'(\d+)\s*Rnd\s*\((\d+)-') #clean time format col into two cols with clean datatypes
    df_fight_oneline['round_total'] = pd.to_numeric(split[0], errors='coerce').astype('Int64')
    df_fight_oneline['round_time_sec'] = pd.to_numeric(split[1], errors='coerce').astype('Int64') * 60
    df_fight_oneline = df_fight_oneline.drop(columns='time_format')
    return df_fight_oneline

def clean_rbr(df_rbr):
    df_rbr['Td %_x'] = df_rbr['Td %_x'].replace('---', np.nan) #clean '---' with NaNs

    for col in df_rbr.columns: #drop duplicate columns error in scraping
        if col.endswith('_y'):
            df_rbr = df_rbr.drop(columns = col)

    columns = ['Sig. str._x', 'Total str._x', 'Td_x', 'Sig. str', #convert cols with two numbers as a string to two seperate cols
               'Head', 'Body', 'Leg', 'Distance', 'Clinch', 'Ground']
    for col in columns:
        if col in df_rbr.columns:
            split = df_rbr[col].str.split(' of ', expand=True)
            df_rbr[col + '_landed'] = pd.to_numeric(split[0], errors = 'coerce',).astype('Int64')
            df_rbr[col + '_attempted'] = pd.to_numeric(split[1], errors = 'coerce',).astype('Int64')
    df_rbr = df_rbr.drop(columns = columns)

    df_rbr.columns = (df_rbr.columns.str.lower().str.replace('.', '_').str.replace(' ', '_') #clean up column names
                      .str.replace('%', 'pct').str.replace('_x', '').str.replace('__', '_'))
    df_rbr = df_rbr.rename(columns={'rev_': 'rev'})

    columns = ['sig_str_pct', 'td_pct']
    for col in columns: #clean cols to integer datatype
        if col in df_rbr.columns:
            df_rbr[col] = pd.to_numeric(df_rbr[col].str.replace('%', ''), errors='coerce').astype('Int64')

    df_rbr['ctrl_secs'] = ((pd.to_numeric(df_rbr['ctrl'].str.split(':', expand=True)[0], errors = 'coerce') * 60) +
                           pd.to_numeric(df_rbr['ctrl'].str.split(':', expand=True)[1], errors = 'coerce')).astype('Int64') #clean time col
    df_rbr = df_rbr.drop(columns = 'ctrl')

    #the notebook leaves these as strings and the csv round-trip made them ints;
    #the db columns are integer so convert here
    for col in ['kd', 'sub_att', 'rev']:
        df_rbr[col] = pd.to_numeric(df_rbr[col], errors='coerce').astype('Int64')

    # fix fight meta-aggregate 'row 0''s missing data (copied from the notebook)
    stat_cols = ['sig_str_landed', 'sig_str_attempted', 'head_landed', 'head_attempted', #cols to be aggregated
                 'body_landed', 'body_attempted', 'leg_landed', 'leg_attempted',
                 'distance_landed', 'distance_attempted', 'clinch_landed', 'clinch_attempted',
                 'ground_landed', 'ground_attempted']

    totals = (df_rbr[df_rbr['round'] > 0] #group aggregation by groups of [fights, fighter] composite key excluding round == 0
              .groupby(['fight_id', 'fighter'])[stat_cols]
              .sum())

    for i, row in df_rbr.iterrows(): #iteratively update values
        if row['round'] == 0:
            for col in stat_cols:
                df_rbr.at[i, col] = totals.loc[(row['fight_id'], row['fighter']), col]

    #postgres cannot take duplicate column names. the 'Sig. str._x' and 'Sig. str'
    #splits both land on sig_str_landed/attempted; pandas called the second copy
    #sig_str_landed.1 on the csv round-trip, which seeded the db as sig_str_landed_1
    seen, cols = {}, []
    for col in df_rbr.columns:
        k = seen.get(col, 0)
        cols.append(col if k == 0 else f'{col}_{k}')
        seen[col] = k + 1
    df_rbr.columns = cols

    return df_rbr

def clean_fighter_stats(df_fighter_stats, fighter_name):
    #single-row version for debut fighters; the name_map merge from the notebook
    #is replaced by the name we already have from the fight page
    df_fighter_stats = df_fighter_stats.replace('--', np.nan) #clean '--' with NaNs

    split = df_fighter_stats['record'].str.split('-', expand=True) #split record col into wins/losses/draws
    df_fighter_stats['wins'] = pd.to_numeric(split[0], errors = 'coerce').astype('Int64')
    df_fighter_stats['losses'] = pd.to_numeric(split[1], errors = 'coerce').astype('Int64')
    df_fighter_stats['draws'] = pd.to_numeric(split[2], errors = 'coerce').astype('Int64')
    df_fighter_stats = df_fighter_stats.drop(columns = 'record')

    columns = ['Str. Acc.:', 'Str. Def:', 'TD Acc.:', 'TD Def.:']
    for col in columns:
        new_col = col.replace(':', '').replace(' ', '_').rstrip('_') + '_pct'
        df_fighter_stats[new_col] = pd.to_numeric(df_fighter_stats[col].str.rstrip('%'), errors='coerce').astype('Int64')
    df_fighter_stats.drop(columns=columns, inplace=True)

    split = df_fighter_stats['Height:'].str.split("' ", expand=True) #convert height to inches
    df_fighter_stats['height_inches'] = ((pd.to_numeric(split[0], errors = 'coerce') * 12) +
                                     pd.to_numeric(split[1].str.rstrip('"'), errors = 'coerce')).astype('Int64')
    df_fighter_stats = df_fighter_stats.drop(columns = 'Height:')

    df_fighter_stats.columns = (df_fighter_stats.columns.str.lower(). #reformating column names
        str.replace(' ', '_').str.replace('.', '').str.replace(':', ''))

    df_fighter_stats['weight'] = pd.to_numeric(df_fighter_stats['weight'].str.replace(' lbs.', '', regex=False), errors='coerce').astype('Int64')
    df_fighter_stats['reach'] = pd.to_numeric(df_fighter_stats['reach'].str.rstrip('"'), errors='coerce').astype('Int64')
    df_fighter_stats['dob'] = pd.to_datetime(df_fighter_stats['dob'], format='%b %d, %Y', errors='coerce') #save col to datetime datatype
    df_fighter_stats['fighter_name'] = fighter_name

    #the blank career-stat label the csv round-trip used to call unnamed_6
    df_fighter_stats = df_fighter_stats.drop(columns=[c for c in df_fighter_stats.columns if c == ''])
    return df_fighter_stats



# ---- db bootstrap and helpers ----

def align_to_table(df, table):
    #reindex to the live table's columns: extras are dropped with a warning,
    #missing columns become NULL. cheap insurance against schema drift
    cols = TABLE_COLS[table]
    extras = [c for c in df.columns if c not in cols]
    if extras:
        print(f'    warning: dropping columns not in {table}: {extras}')
    return df.reindex(columns=cols)

def seed_table(engine, table, df, pk_cols):
    df = df.dropna(subset=pk_cols)
    dupes = df.duplicated(subset=pk_cols).sum()
    if dupes:
        print(f'    dropping {dupes} duplicate {pk_cols} rows before seeding')
        df = df.drop_duplicates(subset=pk_cols, keep='first')
    with engine.begin() as conn:
        df.to_sql(table, conn, if_exists='replace', index=False, chunksize=5000)
        conn.execute(text(f'ALTER TABLE {table} ADD PRIMARY KEY ({", ".join(pk_cols)})'))
    print(f'    seeded {table} with {len(df):,} rows')

def ensure_tables(engine):
    insp = inspect(engine)

    if not insp.has_table('fighter_features'):
        sys.exit('fighter_features is missing -- run load_fighter_features.py first, this script only keeps it current')

    if not insp.has_table('fights'):
        print('fights table missing, seeding from the cleaned csvs (first run against this db)')
        df = pd.read_csv(CLEANING_DIR / 'fighters_fights.csv')
        df['date'] = pd.to_datetime(df['date']).dt.date
        seed_table(engine, 'fights', df, ['fight_id', 'fighter_name'])

    if not insp.has_table('fight_stats'):
        print('fight_stats table missing, seeding from the cleaned csvs')
        df = pd.read_csv(CLEANING_DIR / 'fights.csv')  #pandas mangles the duplicate sig_str cols to .1 on read
        df.columns = [c.replace('.', '_') for c in df.columns]
        seed_table(engine, 'fight_stats', df, ['fight_id', 'fighter', 'round'])

    if not insp.has_table('fighters'):
        print('fighters table missing, seeding from the cleaned csvs')
        df = pd.read_csv(CLEANING_DIR / 'fighters.csv')
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce').dt.date
        seed_table(engine, 'fighters', df, ['fighter_id'])

    insp = inspect(engine)  #re-inspect, tables may have just been created
    for table in ['fights', 'fight_stats', 'fighters', 'fighter_features']:
        TABLE_COLS[table] = [c['name'] for c in insp.get_columns(table)]

    #abort loudly if fighter_features cannot hold what this script computes
    required = ({'fighter', 'date', 'rating', 'rating_deviation', 'volatility', 'all_time_min',
                 'wins', 'losses', 'weight', 'stance', 'dob', 'reach_z', 'height_inches_z', 'age'}
                | {c + '_norm' for c in NORM_COLS} | set(STANCE_COLS))
    missing = required - set(TABLE_COLS['fighter_features'])
    if missing:
        sys.exit(f'fighter_features is missing columns this script fills: {sorted(missing)}')

def last_update_date(engine):
    with engine.connect() as conn:
        d_fights = conn.execute(text('SELECT MAX(date) FROM fights')).scalar()
        d_feats = conn.execute(text('SELECT MAX(date) FROM fighter_features')).scalar()
    if d_fights is None or d_feats is None:
        sys.exit('fights or fighter_features is empty -- instantiate the db before updating it')
    if d_fights != d_feats:
        #diverged tables mean snapshot holes ahead, which would quietly poison
        #every rating that touches them. refuse and make the human look
        sys.exit(f'fights is current through {d_fights} but fighter_features through {d_feats}; '
                 'the instantiation csvs/tables are out of sync, fix that before updating')
    return d_fights



# ---- rating + feature carry (semantics copied from ratings.ipynb and feature_engineering_instantiator.ipynb) ----

def glicko_carry(conn, fighter, event_date):
    '''rating carried INTO the new fight. take the fighter's last stored pre-fight
    snapshot and replay every fight between then and now (normally exactly one).
    opponents stay frozen at their own pre-fight snapshots -- exactly what the
    deepcopies in rating_func do. one deliberate deviation: rating_func scores a
    draw as a win for whichever fighter landed on the odd row of the [::2] slice,
    which is not reconstructible from the db; here a draw is outcome 0 for both.'''
    snap = conn.execute(text(
        'SELECT date, rating, rating_deviation, volatility FROM fighter_features '
        'WHERE fighter = :f AND date < :d ORDER BY date DESC LIMIT 1'),
        {'f': fighter, 'd': event_date}).fetchone()
    if snap is None:
        return 1500.0, 350.0, 0.06 #debut, same defaults as a fresh glicko2.Player()

    player = glicko2.Player(rating=snap.rating, rd=snap.rating_deviation, vol=snap.volatility)

    replays = conn.execute(text(
        'SELECT date, opponent_name, winner_name FROM fights '
        'WHERE fighter_name = :f AND date >= :s AND date < :d ORDER BY date, fight_id'),
        {'f': fighter, 's': snap.date, 'd': event_date}).fetchall()
    if len(replays) == 0:
        print(f'    warning: no fight row behind the {snap.date} snapshot for {fighter}, carrying rating forward unchanged')
    if len(replays) > 1:
        print(f'    warning: replaying {len(replays)} fights for {fighter} (snapshot hole, self-healing)')

    for fight in replays:
        opp = conn.execute(text(
            'SELECT rating, rating_deviation FROM fighter_features WHERE fighter = :o AND date = :d'),
            {'o': fight.opponent_name, 'd': fight.date}).fetchone()
        if opp is None: #same effect as rating_func initializing an unseen opponent
            print(f'    warning: no snapshot for opponent {fight.opponent_name} on {fight.date}, assuming a fresh 1500')
            opp_rating, opp_rd = 1500.0, 350.0
        else:
            opp_rating, opp_rd = opp.rating, opp.rating_deviation
        outcome = 1 if fight.winner_name == fighter else 0
        player.update_player([opp_rating], [opp_rd], [outcome])

    return player.rating, player.rd, player.vol

def stats_carry(conn, fighter, event_date):
    '''prior time-weighted per-minute averages, excluding the current fight.
    summing stat*length over every prior fight and dividing by total minutes is
    exactly the cumsum().shift(1) from feature_engineering_instantiator evaluated
    at the new row, so instantiated and updated snapshots agree by construction.
    only fights that have round-by-round stats count, wins/losses included --
    same inner merge as the instantiator.'''
    stat_select = ', '.join(f'fs.{c}' for c in NORM_COLS)
    df_fights = pd.read_sql(text(
        f'SELECT {stat_select}, f.round_finished, f.round_time_sec, f.stoppage_time_sec, '
        'f.winner_name, f.loser_name '
        'FROM fight_stats fs '
        'JOIN fights f ON f.fight_id = fs.fight_id AND f.fighter_name = fs.fighter '
        'WHERE fs.fighter = :f AND fs.round = 0 AND f.date < :d '
        'ORDER BY f.date'), conn, params={'f': fighter, 'd': event_date})

    out = {'all_time_min': np.nan, 'wins': np.nan, 'losses': np.nan}
    out.update({col + '_norm': np.nan for col in NORM_COLS})
    if len(df_fights) == 0:
        return out #debut: NaN across the board, same as the shift(1) NaNs at instantiation

    num_cols = NORM_COLS + ['round_finished', 'round_time_sec', 'stoppage_time_sec']
    df_fights[num_cols] = df_fights[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0) #same fillna(0) as the instantiator; to_numeric guards against text-typed db columns

    df_fights['fight_length_min'] = (((df_fights['round_finished'] - 1) * df_fights['round_time_sec']) + df_fights['stoppage_time_sec']) / 60

    all_time_min = df_fights['fight_length_min'].sum()
    out['all_time_min'] = all_time_min
    for col in NORM_COLS:
        wsum = (df_fights[col] * df_fights['fight_length_min']).sum()
        out[col + '_norm'] = wsum / all_time_min if all_time_min > 0 else np.nan

    out['wins'] = float((df_fights['winner_name'] == fighter).sum()) #'TIE' rows count for neither
    out['losses'] = float((df_fights['loser_name'] == fighter).sum())
    return out

def fighters_biostats(conn, fighter, fighter_id, page):
    #static biostats for a debut fighter: db first, scrape their page if unseen
    row = conn.execute(text(
        'SELECT weight, reach, height_inches, stance, dob FROM fighters WHERE fighter_id = :i'),
        {'i': fighter_id}).fetchone()
    if row is not None:
        return {'weight': row.weight, 'reach': row.reach, 'height_inches': row.height_inches,
                'stance': row.stance, 'dob': row.dob}

    print(f'    new fighter {fighter}, scraping their page')
    html = single_raw_html_fetcher(f'http://ufcstats.com/fighter-details/{fighter_id}', page)
    df_fighter_stats, _ = single_fighter_scraper(html)
    df_fighter_stats = clean_fighter_stats(df_fighter_stats, fighter)
    df_fighter_stats.insert(0, 'fighter_id', fighter_id)
    df_fighter_stats['dob'] = df_fighter_stats['dob'].dt.date
    align_to_table(df_fighter_stats, 'fighters').to_sql('fighters', conn, if_exists='append', index=False)

    r = df_fighter_stats.iloc[0]
    return {col: (None if pd.isna(r.get(col)) else r.get(col))
            for col in ['weight', 'reach', 'height_inches', 'stance', 'dob']}

def static_carry(conn, fighter, fighter_id, event_date, page):
    '''weight/stance/dob and the weight-class z-scores ride along from the last
    snapshot; only age is recomputed each time. debuts pull biostats from the
    fighters table (scraping their page if needed). one honest approximation:
    the instantiator z-scored reach/height against its full snapshot population,
    which is not stored, so post-instantiation debuts z-score against the
    fighters table for their weight class instead.'''
    prev = conn.execute(text(
        'SELECT weight, stance, dob, reach_z, height_inches_z, '
        'stance_orthodox, stance_southpaw, stance_switch, stance_nan '
        'FROM fighter_features WHERE fighter = :f AND date < :d ORDER BY date DESC LIMIT 1'),
        {'f': fighter, 'd': event_date}).fetchone()

    if prev is not None:
        out = {'weight': prev.weight, 'stance': prev.stance, 'dob': prev.dob,
               'reach_z': prev.reach_z, 'height_inches_z': prev.height_inches_z,
               'stance_orthodox': prev.stance_orthodox, 'stance_southpaw': prev.stance_southpaw,
               'stance_switch': prev.stance_switch, 'stance_nan': prev.stance_nan}
    else:
        bio = fighters_biostats(conn, fighter, fighter_id, page)
        stance = bio['stance'] if bio['stance'] in ('Orthodox', 'Southpaw', 'Switch') else None #rare stances were collapsed to nan at instantiation
        out = {'weight': bio['weight'], 'stance': stance, 'dob': bio['dob'],
               'stance_orthodox': 1.0 if stance == 'Orthodox' else 0.0,
               'stance_southpaw': 1.0 if stance == 'Southpaw' else 0.0,
               'stance_switch': 1.0 if stance == 'Switch' else 0.0,
               'stance_nan': 1.0 if stance is None else 0.0}
        for col in ['reach', 'height_inches']:
            out[col + '_z'] = np.nan
            if bio[col] is not None and bio['weight'] is not None:
                pop = pd.read_sql(text(f'SELECT {col} FROM fighters WHERE weight = :w'),
                                  conn, params={'w': bio['weight']})[col].dropna()
                if len(pop) > 1 and pop.std() > 0:
                    out[col + '_z'] = (bio[col] - pop.mean()) / pop.std()

    #age recomputed every snapshot; same floor((date - dob) / 365.25) as the instantiator
    if out['dob'] is not None and pd.notna(out['dob']):
        out['age'] = int(np.floor((pd.Timestamp(event_date) - pd.Timestamp(out['dob'])).days / 365.25))
    else:
        out['age'] = None
    return out

def build_snapshot(conn, fighter, fighter_id, event_date, page):
    row = {'fighter': fighter, 'date': event_date}
    rating, rd, vol = glicko_carry(conn, fighter, event_date)
    row.update({'rating': rating, 'rating_deviation': rd, 'volatility': vol})
    row.update(stats_carry(conn, fighter, event_date))
    row.update(static_carry(conn, fighter, fighter_id, event_date, page))
    return row



# ---- per-event processing ----

def write_fight(conn, page, fight_id, oneline, df_stats_rows, person_ids, event_name, event_date):
    names = list(dict.fromkeys(df_stats_rows['fighter'])) #both fighters, scrape order preserved
    if len(names) != 2:
        raise ValueError(f'expected 2 fighters in the round table, got {names}')
    fighter_one, fighter_two = names

    #snapshots first: every query inside filters date < event_date, so the state
    #is strictly pre-event no matter what lands in this transaction later
    snapshots = []
    for fighter in (fighter_one, fighter_two):
        exists = conn.execute(text('SELECT 1 FROM fighter_features WHERE fighter = :f AND date = :d'),
                              {'f': fighter, 'd': event_date}).fetchone()
        if exists: #reruns, or the old same-day tournament quirk
            print(f'    warning: snapshot ({fighter}, {event_date}) already exists, keeping the original')
            continue
        snapshots.append(build_snapshot(conn, fighter, person_ids.get(fighter), event_date, page))

    fight_rows = []
    for fighter, opponent in ((fighter_one, fighter_two), (fighter_two, fighter_one)): #two perspectives, fighters_fights schema
        fight_rows.append({
            'fight_id': fight_id,
            'round_finished': int(oneline['round']),
            'winner_name': oneline['winner'],
            'loser_name': oneline['loser'],
            'round_total': None if pd.isna(oneline['round_total']) else int(oneline['round_total']),
            'round_time_sec': None if pd.isna(oneline['round_time_sec']) else int(oneline['round_time_sec']),
            'fighter_id': person_ids.get(fighter),
            'event': event_name,
            'stoppage_time_sec': None if pd.isna(oneline['time_sec']) else int(oneline['time_sec']),
            'opponent_name': opponent,
            'fighter_name': fighter,
            'method_type': METHOD_MAP.get(oneline['method'], oneline['method']),
            'method_specific': None, #lived on the fighter pages; not on the fight page, and nothing downstream reads it
            'date': event_date,
        })

    align_to_table(pd.DataFrame(fight_rows), 'fights').to_sql('fights', conn, if_exists='append', index=False)
    align_to_table(df_stats_rows, 'fight_stats').to_sql('fight_stats', conn, if_exists='append', index=False)
    if snapshots:
        align_to_table(pd.DataFrame(snapshots), 'fighter_features').to_sql('fighter_features', conn, if_exists='append', index=False)
    return len(snapshots)

def process_event(engine, page, event):
    html = single_raw_html_fetcher(event['event_url'], page)
    page_date, fight_urls = event_page_parser(html)
    event_date = event['date']
    if page_date is not None and page_date != event_date: #listing and event page should agree
        print(f'    warning: listing says {event_date}, event page says {page_date}; trusting the event page')
        event_date = page_date
    if len(fight_urls) == 0:
        print('    no fight links yet (event probably has not happened), skipping')
        return 0, 0, 0

    n_new = n_skip = n_fail = 0
    with engine.begin() as conn: #one transaction per event: all of it lands or none of it does
        for url in fight_urls:
            fight_id = id_from_url(url)
            if conn.execute(text('SELECT 1 FROM fights WHERE fight_id = :i LIMIT 1'), {'i': fight_id}).fetchone():
                n_skip += 1
                continue
            try:
                fight_html = single_raw_html_fetcher(url, page)
                df_fight_stats, df_rbr_fight = single_fight_scraper(fight_html) #NCs/overturned crash here on purpose, same discard policy as instantiation
                person_ids = fight_person_ids(fight_html)
                df_fight_stats.insert(0, 'fight_id', fight_id)
                df_rbr_fight.insert(0, 'fight_id', fight_id)
                oneline = clean_fight_oneline(df_fight_stats).iloc[0]
                df_stats_rows = clean_rbr(df_rbr_fight)
                write_fight(conn, page, fight_id, oneline, df_stats_rows, person_ids, event['event_name'], event_date)
                n_new += 1
            except SQLAlchemyError: #db errors are systemic: let the transaction roll back instead of half-committing
                raise
            except Exception as e:
                print(f'    skip fight {url}: {e}')
                save_progress(pd.DataFrame([[url, str(e)]], columns=['url', 'error']), FAILED_CSV)
                n_fail += 1
    return n_new, n_skip, n_fail



# ---- main ----

def parse_iso(s):
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        sys.exit(f'bad date {s!r}, expected YYYY-MM-DD')

def verify_run(engine, total_new, total_skip, total_fail):
    #post-run redundancy check: the tables should agree with each other again
    with engine.connect() as conn:
        d_fights = conn.execute(text('SELECT MAX(date) FROM fights')).scalar()
        d_feats = conn.execute(text('SELECT MAX(date) FROM fighter_features')).scalar()
        n_fights = conn.execute(text('SELECT COUNT(*) FROM fights')).scalar()
        n_feats = conn.execute(text('SELECT COUNT(*) FROM fighter_features')).scalar()

    print(f'\ndone: {total_new} fights added, {total_skip} already in db, {total_fail} failed')
    print(f'    fights: {n_fights:,} rows through {d_fights}')
    print(f'    fighter_features: {n_feats:,} rows through {d_feats}')
    if d_fights != d_feats:
        print('    warning: max dates disagree -- rerun to self-heal, and check the log above')
    if total_fail:
        print(f'    {total_fail} fights logged to {FAILED_CSV} (NCs and overturned fights are expected there)')

def main():
    args = sys.argv[1:]
    today = datetime.date.today()
    if len(args) == 0:
        start_arg, end_date = None, today
    elif len(args) == 1:
        start_arg, end_date = None, parse_iso(args[0])
    elif len(args) == 2:
        start_arg, end_date = parse_iso(args[0]), parse_iso(args[1])
    else:
        sys.exit('usage: python updater.py [end_date] | [start_date end_date]   (dates are YYYY-MM-DD)')

    engine = create_engine(DB_URL)
    ensure_tables(engine)
    last_date = last_update_date(engine)
    print(f'db is current through {last_date}')

    #the ratings chain cannot tolerate a gap, so the window never starts after
    #the db state. starting earlier is harmless: existing fights get skipped
    start_date = last_date
    if start_arg is not None:
        if start_arg > last_date:
            print(f'warning: requested start {start_arg} is after the db state, clamping to {last_date} so no gap corrupts the chain')
        else:
            start_date = start_arg
    if end_date < start_date:
        sys.exit(f'end date {end_date} is before start date {start_date}, nothing to do')
    if end_date > today:
        print('note: events after today have no results yet and will be skipped when empty')

    total_new = total_skip = total_fail = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        events = completed_events_scraper(page)
        #start date inclusive: if a run died mid-day, the surviving events of that
        #day are deduped fight by fight and the rolled-back one gets redone
        window = [e for e in events if start_date <= e['date'] <= end_date]
        print(f'{len(window)} events in window {start_date} -> {end_date}')

        for event in window: #oldest first, so every rating update sees a settled past
            print(f"{event['event_name']} ({event['date']})")
            n_new, n_skip, n_fail = process_event(engine, page, event)
            print(f'    {n_new} fights added, {n_skip} already in db, {n_fail} failed')
            total_new += n_new; total_skip += n_skip; total_fail += n_fail

        page.close()
        browser.close()

    verify_run(engine, total_new, total_skip, total_fail)

if __name__ == '__main__':
    main()
