import os
import sys
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import glicko2
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

'''
updater pipeline: catch the neon database up to a target date, with receipts
    1) ask fighter_features for the last date it knows about (MAX(date))
    2) scrape ufcstats.com completed events from that date up to the target date,
       least recent first, skipping any fight_id we already have
    3) clean the new fights with the same code as 2. data_cleaning (copied text,
       not imported -- the notebooks stay untouched)
    4) recompute glicko ratings + features over the FULL history (old csvs + new
       appends) with the same code as ratings.ipynb / feature_engineering_instantiator,
       so one pipeline run can never drift from the instantiation
    5) rebuild fighter_features on neon behind a backup-table swap
failsafes (belt, suspenders, second belt)
    - raw scrapes append to updater_data/*.csv first; cleaned/derived files are
      REGENERATED from those logs every run, so a crash anywhere = just rerun
    - already-scraped and known-failed fight_ids are skipped (resume for free)
    - every url gets retries; fights that still fail land in a failed csv like
      the original scraper did, and are excluded rather than half-included
    - pairing/column/dupe guards abort loudly before anything touches the db
    - db swap keeps the old table as fighter_features_backup (rollback = rename it back)
usage
    pip install pandas numpy sqlalchemy psycopg2-binary playwright beautifulsoup4 glicko2 scikit-learn   (first time only)
    python updater.py 2026-07-23        (target date; defaults to today if omitted)
    set UFC_DB_URL to point at neon -- keep it in the environment, never in this file
'''

DB_URL = os.environ.get('UFC_DB_URL', 'postgresql+psycopg2://alejandrogomez-paz@localhost:5432/ufc')
TABLE = 'fighter_features'

ROOT = Path(__file__).resolve().parent.parent
SCRAPE_DIR = ROOT / '1. data_scraping'
CLEAN_DIR = ROOT / '2. data_cleaning'
DATA_DIR = Path(__file__).resolve().parent / 'updater_data'   # append-only logs + derived files live here

# converged tau from ratings.ipynb; static, do not re-optimize on updates
OPTIMAL_TAU = 0.10040325212871097

# exact column order of 2. data_cleaning/fighters_fights.csv, so concat lines up
FF_COLS = ['fight_id', 'round_finished', 'winner_name', 'loser_name', 'round_total',
           'round_time_sec', 'fighter_id', 'event', 'stoppage_time_sec', 'opponent_name',
           'fighter_name', 'method_type', 'method_specific', 'date']


#---------------------------------------------------------------------------------
# scraping -- copied from 1. data_scraping/CODE_DATA_SCRAPING.ipynb (keep in sync)
#---------------------------------------------------------------------------------

def single_raw_html_fetcher(url, page):
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_selector("body .l-page, table, tbody", timeout=15000)
    except:
        pass
    page.wait_for_timeout(200)
    html = page.content()
    return html

def id_from_url(url):
    return url.rstrip('/').split('/')[-1]

def save_progress(df, filename):
    df.to_csv( filename, mode="a", header=not os.path.exists(filename), index=False )

def fight_url_scraper(event_urls_list, page):

    fight_urls = []
    for url in event_urls_list:
        html = single_raw_html_fetcher(url, page)
        soup = BeautifulSoup(html, "html.parser")


        event_soup = soup.find('tbody', class_ = "b-fight-details__table-body")
        row_soup = event_soup.find_all('tr')

        for row in row_soup:
            link = row.get('data-link')
            if link:
                fight_urls.append(link)

    fight_urls = list(set(fight_urls)) # precautionary dedupe after iteration...
    return fight_urls

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


#---------------------------------------------------------------------------------
# scraping -- updater-only additions (events with dates, person links, retries)
#---------------------------------------------------------------------------------

def fetch_with_retry(url, page, tries = 3):
    for attempt in range(tries): #failsafe: ufcstats hiccups, so every url gets three chances
        try:
            return single_raw_html_fetcher(url, page)
        except Exception as e:
            if attempt == tries - 1:
                raise
            page.wait_for_timeout(1500)

def completed_events_scraper(page):
    #same table as events_page_scraper in CODE_DATA_SCRAPING, but keeping the date span too
    url = 'http://ufcstats.com/statistics/events/completed?page=all'
    html = fetch_with_retry(url, page)
    soup = BeautifulSoup(html, "html.parser")

    soup = soup.find('table', class_ = "b-statistics__table-events")
    row_soup = soup.find_all('tr', class_ = 'b-statistics__table-row')

    events = []
    for row in row_soup:
        a_tag = row.find('a', class_ = 'b-link b-link_style_black')
        date_tag = row.find('span', class_ = 'b-statistics__date')
        if a_tag and a_tag.get('href'):
            events.append([a_tag['href'], a_tag.text.strip(),
                           pd.to_datetime(date_tag.text.strip(), errors = 'coerce') if date_tag else pd.NaT])

    df_events = pd.DataFrame(events, columns = ['event_url', 'event_name', 'event_date'])
    return df_events

def event_page_date(event_url, page):
    #failsafe: second, independent source for the event date if the listing span fails
    html = fetch_with_retry(event_url, page)
    soup = BeautifulSoup(html, "html.parser")
    for li in soup.find_all('li', class_ = 'b-list__box-list-item'):
        if 'Date:' in li.text:
            return pd.to_datetime(li.text.replace('Date:', '').strip(), errors = 'coerce')
    return pd.NaT

def fight_person_links(html):
    #the two fighter names + fighter-details urls off a fight page (same class the winner/loser scrape uses)
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all('a', class_ = 'b-link b-fight-details__person-link')
    return [(a.text.strip(), a['href']) for a in links]


#---------------------------------------------------------------------------------
# cleaning -- copied from 2. data_cleaning/data_cleaning.ipynb (keep in sync)
#---------------------------------------------------------------------------------

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

    #fix fight meta-aggregate 'row 0''s missing data (same loop as the cleaning notebook)
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

    return df_rbr

def clean_fighter_stats(df_fighter_stats):

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

    #the notebook mapped fighter_name in from the fighter-fights table; here the name rode
    #in on the scrape (person link text), so it is already a column and no merge is needed
    for col in df_fighter_stats.columns: #same blank-column artifact the notebook dropped as 'unnamed_6'
        if col.startswith('unnamed'):
            df_fighter_stats = df_fighter_stats.drop(columns = col)

    df_fighter_stats['dob'] = pd.to_datetime(df_fighter_stats['dob'], format='%b %d, %Y', errors='coerce') #save col to datetime datatype

    return df_fighter_stats


#---------------------------------------------------------------------------------
# ratings -- copied from 3. feature_engineering_and_model_training/ratings.ipynb
#---------------------------------------------------------------------------------

def rating_func(df_fighters_fights):

    rating_dict = {} # [fighter, date]: glicko rating
    initialized_fighters = set()

    for i, row in df_fighters_fights.iterrows():
        fighter, opponent, date = row['fighter_name'], row['opponent_name'], row['date']
        if fighter not in initialized_fighters:
            rating_dict[(fighter, date)] = glicko2.Player()
            initialized_fighters.add(fighter)
        if opponent not in initialized_fighters:
            rating_dict[(opponent, date)] = glicko2.Player()
            initialized_fighters.add(opponent)

        last_date = max([d for (f, d) in rating_dict.keys() if f == fighter])
        opp_last_date = max([d for (f, d) in rating_dict.keys() if f == opponent])
        fighter_object = copy.deepcopy(rating_dict[(fighter, last_date)])
        opponent_object = copy.deepcopy(rating_dict[(opponent, opp_last_date)])

        # next two lines in order to freeze variables form objects; makes logic less messy and less variable tracking :)
        fighter_rating, fighter_rd = rating_dict[(fighter, last_date)].rating, rating_dict[(fighter, last_date)].rd
        opponent_rating, opponent_rd = rating_dict[(opponent, opp_last_date)].rating, rating_dict[(opponent, opp_last_date)].rd

        if fighter == row['winner_name']:
            fighter_object.update_player([opponent_rating], [opponent_rd], [1])
            rating_dict[(fighter, date)] = fighter_object

            opponent_object.update_player([fighter_rating], [fighter_rd], [0])
            rating_dict[(opponent, date)] = opponent_object
        else:
            fighter_object.update_player([opponent_rating], [opponent_rd], [0])
            rating_dict[(fighter, date)] = fighter_object

            opponent_object.update_player([fighter_rating], [fighter_rd], [1])
            rating_dict[(opponent, date)] = opponent_object


    df_ratings = pd.DataFrame(
        [(f, d, p.rating, p.rd, p.vol) for (f, d), p in rating_dict.items()],
        columns=['fighter', 'prior_to_date', 'rating', 'rating_deviation', 'volatility'])

    df_ratings = df_ratings.sort_values(['fighter', 'prior_to_date'], ascending = True)

    # pre-fight rating: the rating carried INTO each fight (previous fight's result)
    df_ratings['rating'] = df_ratings.groupby('fighter')['rating'].shift(1).fillna(1500)
    df_ratings['rating_deviation']     = df_ratings.groupby('fighter')['rating_deviation'].shift(1).fillna(350)
    df_ratings['volatility'] = df_ratings.groupby('fighter')['volatility'].shift(1).fillna(0.06)

    return df_ratings


#---------------------------------------------------------------------------------
# features -- copied from feature_engineering_instantiator.ipynb (reads the frames
# passed in instead of the csvs; everything else is the same text)
#---------------------------------------------------------------------------------

def build_fighter_database(df_fighters_fights, df_rbr, df_fighters, df_ratings):

    cols1 = ['fighter_id']
    df_fighters_fights = df_fighters_fights.drop(columns = cols1)
    df_fighters_fights = df_fighters_fights.sort_values('date', ascending = True)
    df_fighters_fights = df_fighters_fights.rename(columns={'fighter_name': 'fighter'})

    df_rbr_agg = df_rbr[df_rbr['round'] == 0]
    df_fights = pd.merge(df_rbr_agg, df_fighters_fights, on=['fight_id', 'fighter'])
    df_fights = df_fights.fillna(0)



    # clean and merge fighter static data with ratings on composite primary key
    cols2 = ['slpm',  'sapm', 'td_avg', 'sub_avg', 'wins', 'losses', 'draws', 'str_acc_pct',  'str_def_pct', 'td_acc_pct', 'td_def_pct']
    df_fighters = df_fighters.drop(columns = cols2)

    df_ratings = df_ratings.rename(columns={'prior_to_date': 'date'})

    n = df_fighters['fighter_name'].duplicated().sum()
    print(f"{n} rows would be dropped")

    df_fighters_u = df_fighters.drop_duplicates('fighter_name', keep='first')
    df = pd.merge(df_ratings, df_fighters_u,
                      left_on='fighter', right_on='fighter_name',
                      how='left', validate='m:1')




    # time-weighted running average of each stat, EXCLUDING the current fight (no leakage)
    df_fights['fight_length_min'] = (((df_fights['round_finished'] - 1) * df_fights['round_time_sec']) + df_fights['stoppage_time_sec']) / 60

    df_fights = df_fights.sort_values(['fighter', 'date']).reset_index(drop=True)
    g = df_fights.groupby('fighter')

    agg = df_fights[['fighter', 'date']].copy()
    agg['all_time_min'] = g['fight_length_min'].cumsum().shift(1)
    agg.loc[g.cumcount() == 0, 'all_time_min'] = np.nan

    cols = ['sig_str_pct', 'td_pct', 'sub_att', 'rev', 'sig_str_landed', 'sig_str_attempted',
            'total_str_landed', 'total_str_attempted', 'td_landed', 'td_attempted',
            'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
            'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
            'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted', 'ctrl_secs']

    for col in cols:
        wsum = df_fights[col] * df_fights['fight_length_min']
        prior_wsum = wsum.groupby(df_fights['fighter']).cumsum().shift(1)
        prior_wsum[g.cumcount() == 0] = np.nan
        agg[col + '_norm'] = prior_wsum / agg['all_time_min']

    agg['wins'] = df_fights.assign(w=(df_fights['fighter'] == df_fights['winner_name'])).groupby('fighter')['w'].cumsum().shift(1)
    agg['losses'] = df_fights.assign(l=(df_fights['fighter'] == df_fights['loser_name'])).groupby('fighter')['l'].cumsum().shift(1)

    df = df.merge(agg, on=['fighter', 'date'], how='left')



    # static fighter biostats normalized by weightclass
    cols_to_norm_by_weightclass = ['reach', 'height_inches']
    for col in cols_to_norm_by_weightclass:
        grp = df.groupby('weight')[col]
        df[col + '_z'] = (df[col] - grp.transform('mean')) / grp.transform('std')
    df = df.drop(columns = cols_to_norm_by_weightclass)



    # stance to numeric through One Hot Encoding
    counts = df['stance'].value_counts()
    rare = counts[counts < df['stance'].isna().sum()].index
    df['stance'] = df['stance'].where(~df['stance'].isin(rare), np.nan)
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = enc.fit_transform(df[['stance']])
    df[enc.get_feature_names_out(['stance'])] = encoded


    df = df.drop(columns = ['fighter_name', 'fighter_id'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['date'] = pd.to_datetime(df['date'])
    age_years = (df['date'] - df['dob']).dt.days / 365.25
    df['age'] = np.floor(pd.to_numeric(age_years, errors='coerce')).astype('Int64')

    df_u = df[~df.duplicated(['fighter', 'date'], keep=False)]

    return df_u


#---------------------------------------------------------------------------------
# updater orchestration
#---------------------------------------------------------------------------------

def last_db_date(engine):
    #primary source of truth: the table itself. cross-checked against the local csv below
    with engine.connect() as conn:
        db_max = conn.execute(text(f"SELECT MAX(date) FROM {TABLE}")).scalar()
    if db_max is None:
        raise SystemExit(f"'{TABLE}' is empty or missing -- run load_fighter_features.py once before updating.")

    csv_max = pd.to_datetime(pd.read_csv(CLEAN_DIR / 'fighters_fights.csv')['date']).max()
    db_max = pd.to_datetime(db_max)
    if csv_max.date() != db_max.date():
        print(f"note: db max date ({db_max.date()}) != fighters_fights.csv max date ({csv_max.date()}); "
              "using the earlier one so nothing gets skipped")
    return min(db_max, csv_max)

def read_if_exists(path, **kwargs):
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None

def known_fight_ids():
    #failsafe: three independent sources of 'already have it' -- cleaned history,
    #this updater's own raw log, and every failed-fights log (NC/overturned etc.)
    known = set(pd.read_csv(CLEAN_DIR / 'fighters_fights.csv')['fight_id'])

    for path in [DATA_DIR / 'raw_fight_oneline_stats.csv',
                 SCRAPE_DIR / 'failed_fights.csv', SCRAPE_DIR / 'missing_failed_fights.csv',
                 DATA_DIR / 'failed_fights.csv']:
        df = read_if_exists(path)
        if df is not None:
            if 'fight_id' in df.columns:
                known |= set(df['fight_id'])
            else:
                known |= set(df['url'].str.rstrip('/').str.split('/').str[-1])
    return known

def known_fighter_names():
    names = set(pd.read_csv(CLEAN_DIR / 'fighters.csv')['fighter_name'].dropna())
    df = read_if_exists(DATA_DIR / 'raw_fighter_stats.csv')
    if df is not None:
        names |= set(df['fighter_name'].dropna())
    return names

def scrape_new_fights(last_date, target_date):
    #sweep completed events oldest-first, scraping only fights we do not have yet.
    #everything lands in append-only raw logs first, so a crash mid-run loses nothing
    known = known_fight_ids()
    names = known_fighter_names()
    n_new = 0

    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    df_events = completed_events_scraper(page)
    for i, row in df_events.iterrows(): #failsafe: re-derive missing dates from the event page itself
        if pd.isna(row['event_date']):
            df_events.at[i, 'event_date'] = event_page_date(row['event_url'], page)

    #>= last_date on purpose: a second event on the last-updated day would slip through '>'
    #the fight_id check keeps the overlap from double-scraping anything
    df_events = df_events[(df_events['event_date'] >= last_date) & (df_events['event_date'] <= target_date)]
    df_events = df_events.sort_values('event_date', ascending = True) #least recent first
    print(f"{len(df_events)} completed event(s) in window {last_date.date()} -> {target_date.date()}")

    for _, event in df_events.iterrows():
        event_fight_urls = fight_url_scraper([event['event_url']], page)
        fresh = [url for url in event_fight_urls if id_from_url(url) not in known]
        print(f"{event['event_name']} ({event['event_date'].date()}): {len(fresh)} new / {len(event_fight_urls)} fights")

        for url in fresh:
            try:
                html = fetch_with_retry(url, page)
                persons = fight_person_links(html)
                if len(persons) != 2:
                    raise ValueError(f"expected 2 fighters on fight page, got {len(persons)}")

                df3, df4 = single_fight_scraper(html)
                fid = id_from_url(url)
                df3.insert(0, 'fight_id', fid)
                df4.insert(0, 'fight_id', fid)

                #debut fighters: grab their page now so biostats exist at recompute time
                for name, fighter_url in persons:
                    if name not in names:
                        print(f"    new fighter: {name}")
                        df_stats, _ = single_fighter_scraper(fetch_with_retry(fighter_url, page))
                        df_stats.insert(0, 'fighter_id', id_from_url(fighter_url))
                        df_stats['fighter_name'] = name
                        save_progress(df_stats, DATA_DIR / 'raw_fighter_stats.csv')
                        names.add(name)

                #meta rides in its own log so fighters_fights rows can be rebuilt after any crash
                (f1_name, f1_url), (f2_name, f2_url) = persons
                df_meta = pd.DataFrame([[fid, event['event_name'], event['event_date'].strftime('%Y-%m-%d'),
                                         f1_name, id_from_url(f1_url), f2_name, id_from_url(f2_url)]],
                                       columns = ['fight_id', 'event', 'date',
                                                  'fighter_1', 'fighter_1_id', 'fighter_2', 'fighter_2_id'])

                save_progress(df3, DATA_DIR / 'raw_fight_oneline_stats.csv')
                save_progress(df4, DATA_DIR / 'raw_fights_roundbyround.csv')
                save_progress(df_meta, DATA_DIR / 'fight_meta.csv')
                known.add(fid)
                n_new += 1
            except Exception as e:
                print(f"skip fight {url}: {e}")
                df_fail = pd.DataFrame([[url, str(e)]], columns=['url', 'error'])
                save_progress(df_fail, DATA_DIR / 'failed_fights.csv')

    page.close()
    browser.close()
    p.stop()
    return n_new

def rebuild_appends():
    #regenerated (not appended) every run from the raw logs -> rerunning is always safe.
    #round-trips through csv on purpose: pandas mangles the duplicate sig-str columns on
    #read exactly like it did for the original fights.csv, so the frames line up
    raw_oneline = read_if_exists(DATA_DIR / 'raw_fight_oneline_stats.csv')
    raw_rbr = read_if_exists(DATA_DIR / 'raw_fights_roundbyround.csv')
    meta = read_if_exists(DATA_DIR / 'fight_meta.csv')
    if raw_oneline is None or raw_rbr is None or meta is None:
        return None, None, None

    raw_oneline = raw_oneline.drop_duplicates('fight_id') #failsafe: double-append protection
    raw_rbr = raw_rbr.drop_duplicates(['fight_id', 'Fighter', 'Round'])
    meta = meta.drop_duplicates('fight_id')

    df_oneline = clean_fight_oneline(raw_oneline)
    df_rbr_new = clean_rbr(raw_rbr)

    #build the two per-fighter perspective rows the old fighter-page scrape used to provide.
    #method_specific lived on the fighter page only, so it stays NaN here (nothing downstream reads it)
    ff_rows = []
    df_oneline = df_oneline.merge(meta, on = 'fight_id')
    df_oneline = df_oneline.sort_values(['date', 'fight_id'], ascending = True)
    for _, row in df_oneline.iterrows():
        for me, me_id, them in [('fighter_1', 'fighter_1_id', 'fighter_2'),
                                ('fighter_2', 'fighter_2_id', 'fighter_1')]:
            ff_rows.append([row['fight_id'], row['round'], row['winner'], row['loser'],
                            row['round_total'], row['round_time_sec'], row[me_id], row['event'],
                            row['time_sec'], row[them], row[me], row['method'], np.nan, row['date']])
    df_ff_new = pd.DataFrame(ff_rows, columns = FF_COLS)

    raw_fighters = read_if_exists(DATA_DIR / 'raw_fighter_stats.csv')
    df_fighters_new = None
    if raw_fighters is not None:
        df_fighters_new = clean_fighter_stats(raw_fighters.drop_duplicates('fighter_id'))

    #csv round-trip (see note above), also doubles as an audit trail of what this updater added
    df_ff_new.to_csv(DATA_DIR / 'fighters_fights_appended.csv', index=False)
    df_rbr_new.to_csv(DATA_DIR / 'fights_appended.csv', index=False)
    if df_fighters_new is not None:
        df_fighters_new.to_csv(DATA_DIR / 'fighters_appended.csv', index=False)

    df_ff_new = pd.read_csv(DATA_DIR / 'fighters_fights_appended.csv')
    df_rbr_new = pd.read_csv(DATA_DIR / 'fights_appended.csv')
    df_fighters_new = read_if_exists(DATA_DIR / 'fighters_appended.csv')
    return df_ff_new, df_rbr_new, df_fighters_new

def recompute(df_ff_new, df_rbr_new, df_fighters_new):
    #full-history recompute with the instantiation code = zero drift between updates
    #and instantiation, and multi-event gaps / debuts / ties all come out identical
    df_ff = pd.read_csv(CLEAN_DIR / 'fighters_fights.csv')
    df_rbr = pd.read_csv(CLEAN_DIR / 'fights.csv')
    df_fighters = pd.read_csv(CLEAN_DIR / 'fighters.csv')

    if list(df_rbr_new.columns) != list(df_rbr.columns):
        raise SystemExit("Aborting: fights_appended.csv columns do not match fights.csv -- "
                         "ufcstats layout probably changed, fix the scraper before touching the db.")

    df_ff = pd.concat([df_ff, df_ff_new], ignore_index=True)
    df_rbr = pd.concat([df_rbr, df_rbr_new], ignore_index=True)
    if df_fighters_new is not None:
        df_fighters = pd.concat([df_fighters, df_fighters_new], ignore_index=True)

    #failsafe: ratings and features assume two adjacent rows per fight ([::2] one-per-fight trick)
    if len(df_ff) % 2 or (df_ff['fight_id'].values[::2] != df_ff['fight_id'].values[1::2]).any():
        raise SystemExit("Aborting: fighters_fights pairing broke (rows must come in adjacent pairs per fight).")

    #same prep text as ratings.ipynb
    df = df_ff.copy()
    cols = ['round_finished', 'round_time_sec', 'round_total', 'round_time_sec', 'fighter_id', 'event', 'stoppage_time_sec', 'method_type', 'method_specific']
    df = df.drop(columns = cols)
    df = df[::2]
    df = df.sort_values('date', ascending = True)

    print(f"computing glicko ratings over {len(df):,} fights (same loop as ratings.ipynb, takes a few minutes)...")
    glicko2.Player._tau = OPTIMAL_TAU
    df_ratings = rating_func(df)

    df_u = build_fighter_database(df_ff, df_rbr, df_fighters, df_ratings)
    df_u.to_csv(DATA_DIR / 'fighter_database_updated.csv') #index kept, load reads index_col=0 like load_fighter_features

    #failsafe: every new fight should have produced two snapshot rows; name mismatches show up here
    got = set(zip(df_u['fighter'], df_u['date'].dt.strftime('%Y-%m-%d')))
    missing = [(f, d) for f, d in zip(df_ff_new['fighter_name'], df_ff_new['date']) if (f, d) not in got]
    if missing:
        print(f"warning: {len(missing)} expected (fighter, date) rows missing from rebuild "
              f"(dupe-drop or a name mismatch between fight page and fighter page): {missing[:5]}")

    return df_u

def load_to_db(engine):
    #same load as load_fighter_features.py, but staged through a swap so the live
    #table is never half-written and the old one survives as _backup
    df = pd.read_csv(DATA_DIR / 'fighter_database_updated.csv', index_col=0)

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
            f"Aborting: {dupes} duplicate (fighter, date) rows in rebuild. "
            "Fix the data, then rerun."
        )

    with engine.connect() as conn:
        old_n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
        old_max = conn.execute(text(f"SELECT MAX(date) FROM {TABLE}")).scalar()

    #failsafe: an update should only ever grow the table and move the clock forward
    if len(df) < old_n:
        raise SystemExit(f"Aborting: rebuild has {len(df):,} rows but '{TABLE}' already has {old_n:,}. "
                         "Refusing to shrink the table -- inspect updater_data/ first.")
    if df["date"].max() < old_max:
        raise SystemExit(f"Aborting: rebuild max date {df['date'].max()} is behind the db's {old_max}.")

    with engine.begin() as conn:
        df.to_sql(f"{TABLE}_new", conn, if_exists="replace", index=False)
        conn.execute(text(f"ALTER TABLE {TABLE}_new ADD PRIMARY KEY (fighter, date)"))

    with engine.begin() as conn: #single transaction, so the live table swaps atomically
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}_backup"))
        conn.execute(text(f"ALTER TABLE {TABLE} RENAME TO {TABLE}_backup"))
        conn.execute(text(f"ALTER TABLE {TABLE}_new RENAME TO {TABLE}"))

    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
        new_max = conn.execute(text(f"SELECT MAX(date) FROM {TABLE}")).scalar()
    print(f"Loaded {n:,} rows into '{TABLE}' (+{n - old_n:,}), max date {old_max} -> {new_max}.")
    print(f"old table kept as '{TABLE}_backup' -- rollback is just renaming it back.")


def main():
    if len(sys.argv) > 2:
        sys.exit('usage: python updater.py [YYYY-MM-DD]   (target date, defaults to today)')
    target_date = pd.to_datetime(sys.argv[1]) if len(sys.argv) == 2 else pd.Timestamp.today().normalize()
    if pd.isna(target_date):
        sys.exit(f"could not parse '{sys.argv[1]}' as a date")

    os.makedirs(DATA_DIR, exist_ok=True)
    engine = create_engine(DB_URL)

    last_date = last_db_date(engine)
    print(f"db last updated {last_date.date()}, target {target_date.date()}")
    if target_date < last_date:
        sys.exit('target date is before the last update -- nothing to do')

    n_new = scrape_new_fights(last_date, target_date)
    print(f"scraped {n_new} new fight(s) this run")

    df_ff_new, df_rbr_new, df_fighters_new = rebuild_appends()
    if df_ff_new is None or df_ff_new.empty:
        sys.exit('no new fights in the logs -- database already up to date, nothing to load')

    #no-op guard: nothing scraped this run and every logged fight is strictly older than the
    #db clock (i.e. already folded in by a previous successful run) -> skip the recompute
    if n_new == 0 and pd.to_datetime(df_ff_new['date']).max() < last_date:
        sys.exit('logs contain nothing newer than the db -- already up to date')

    recompute(df_ff_new, df_rbr_new, df_fighters_new)
    load_to_db(engine)


if __name__ == '__main__':
    main()
