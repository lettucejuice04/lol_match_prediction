import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from itertools import combinations 
from sqlalchemy import text, create_engine
from pangres import upsert
import psycopg2
import ast
import numpy as np
from sqlalchemy.dialects.postgresql import JSONB


load_dotenv()
API_KEY = os.environ.get('API_KEY')

db_username=os.environ.get('db_username')
db_password=os.environ.get('db_password')
db_host=os.environ.get('db_host')
db_port=os.environ.get('db_port')
db_name=os.environ.get('db_name')

champion_list = [
    "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Amumu", "Anivia", "Annie", "Aphelios", "Ashe",
    "AurelionSol", "Azir", "Bard", "Belveth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn",
    "Camille", "Cassiopeia", "Chogath", "Corki", "Darius", "Diana", "DrMundo", "Draven", "Ekko",
    "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", "Garen",
    "Gnar", "Gragas", "Graves", "Gwen", "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia",
    "Ivern", "Janna", "JarvanIV", "Jax", "Jayce", "Jhin", "Jinx", "KSante", "Kaisa", "Kalista",
    "Karma", "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Khazix", "Kindred",
    "Kled", "KogMaw", "Leblanc", "LeeSin", "Leona", "Lillia", "Lissandra", "Lucian", "Lulu",
    "Lux", "Malphite", "Malzahar", "Maokai", "Milio", "MissFortune", "Mordekaiser", "Morgana",
    "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu",
    "Olaf", "Orianna", "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus",
    "RekSai", "Rell", "Renata", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira",
    "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir",
    "Skarner", "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "TahmKench", "Taliyah",
    "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "TwistedFate",
    "Twitch", "Udyr", "Urgot", "Varus", "Vayne", "Veigar", "Velkoz", "Vex", "Vi", "Viego", "Viktor",
    "Vladimir", "Volibear", "Warwick", "MonkeyKing", "Xayah", "Xerath", "XinZhao", "Yasuo", "Yone",
    "Yorick", "Yuumi", "Zac", "Zed", "Zeri", "Ziggs", "Zilean", "Zoe", "Zyra",
    "Ambessa", "Aurora", "Yunara", "Mel", "MasterYi"
]

REGION_TO_ROUTING = {
    "na1": "americas",
    "euw1": "europe",
    "kr": "asia",
    # add others if needed
}

def fetch_with_backoff(url, max_retries=10):
    """Fetch a URL with retry and backoff logic for rate limits and network errors."""
    for _ in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 1))
                print(f"Rate limit hit, waiting {wait}s")
                time.sleep(wait + 1)
            elif 500 <= response.status_code < 600:
                print(f"Server error {response.status_code}, retrying in 5s")
                time.sleep(5)
            else:
                print(f"Error {response.status_code}: {response.text}")
                return response
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}, retrying in 10s")
            time.sleep(10)
    return None

def get_puuid(game_name, tag_line, region="na1"):
    """Get puuid from Riot ID, region = platform routing (na1, euw1, kr, etc.)"""
    link = f"https://{REGION_TO_ROUTING.get(region, 'americas')}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}?api_key={API_KEY}"
    response = fetch_with_backoff(link)
    return response.json()['puuid']

def get_chall_ladder(region="na1"):
    """Get challenger ladder DataFrame for given platform region"""
    root = f"https://{region}.api.riotgames.com/lol/league/v4"
    url = f"{root}/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key={API_KEY}"
    response = fetch_with_backoff(url)
    return pd.DataFrame(response.json()["entries"])

def get_chall_ladder_puuid(region="na1"):
    """Get challenger ladder puuids"""
    chall_df = get_chall_ladder(region)
    return chall_df[["puuid"]]

def get_match_history(puuid, region="na1", number=20):
    """Get match history for a player puuid. Uses correct regional routing."""
    routing = REGION_TO_ROUTING.get(region, "americas")
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=50&api_key={API_KEY}"
    match_ids = fetch_with_backoff(url).json()

    ranked_ids = []
    for match_id in match_ids:
        match_url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={API_KEY}"
        match_data = fetch_with_backoff(match_url).json()
        queue_id = match_data.get("info", {}).get("queueId", None)
        if queue_id in [420, 440]:
            ranked_ids.append(match_id)
        if len(ranked_ids) >= number:
            break
    return pd.DataFrame(ranked_ids, columns=["match_id"])

def combine_match_histories(puuid_df, region="na1", number=20):
    """Combine match histories for multiple puuids into a single DataFrame"""
    all_matches = []
    for puuid in puuid_df['puuid']:
        match_history = get_match_history(puuid, region=region, number=number)
        all_matches.append(match_history)
    return pd.concat(all_matches, ignore_index=True)

def get_champion_id_name_map():
    """Maps champion id to name"""
    version = fetch_with_backoff("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    data = fetch_with_backoff(url).json()["data"]
    id_to_name = {int(info["key"]): name for name, info in data.items()}
    return id_to_name

def get_champ_name_and_winner(match_id, region="na1"):
    """Get champion names for both teams and the winning side for a given match_id"""
    routing = REGION_TO_ROUTING.get(region, "americas")
    data = fetch_with_backoff(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={API_KEY}").json()
    participants = data["info"]["participants"]
    blue_team_ids = [p["championId"] for p in participants if p["teamId"] == 100]
    red_team_ids = [p["championId"] for p in participants if p["teamId"] == 200]

    champ_id_map = get_champion_id_name_map()
    blue_team_names = [champ_id_map.get(champ_id, "Unknown") for champ_id in blue_team_ids]
    red_team_names = [champ_id_map.get(champ_id, "Unknown") for champ_id in red_team_ids]

    winner_side = None
    for team in data["info"]["teams"]:
        if team["win"]:
            winner_side = "blue" if team["teamId"] == 100 else "red"
            break

    return {
        "match_id": match_id,
        "blue_team": blue_team_names,
        "red_team": red_team_names,
        "winner": winner_side
    }

def get_all_champs_with_winner(match_df, region="na1"):
    """Get champion names and winners for all matches in match_df"""
    all_champ_data = []
    for match_id in match_df["match_id"]:
        try:
            champ_info = get_champ_name_and_winner(match_id, region=region)
            all_champ_data.append(champ_info)
        except Exception as e:
            print(f"error for match {match_id}: {e}")
    df = pd.DataFrame(all_champ_data)
    return df.drop(columns=["match_id"])

def create_db_connection_string(db_username, db_password, db_host, db_port, db_name):
    connection_url = "postgresql+psycopg2://"+db_username+":"+db_password+"@"+db_host+":"+db_port+"/"+db_name
    return connection_url

def create_db_engine():
    """
    Creates and returns a SQLAlchemy database engine using the connection string.
    """
    conn_url = create_db_connection_string(db_username, db_password, db_host, db_port, db_name)
    return create_engine(conn_url, pool_recycle=3600)

def submit_to_sql_and_fetch(df, schema, table_name):
    """
    Submits a DataFrame to a SQL database and retrieves it back as 'df'.

    Parameters:
        df (pd.DataFrame): The DataFrame to submit.
        schema (str): Schema name.
        table_name (str): Table name.

    Returns:
        pd.DataFrame: The DataFrame retrieved from the SQL database.
    """
    # Reset index but don't force a uid column unless you want one
    df = df.reset_index(drop=True)

    db_engine = create_db_engine()

    # Submit the DataFrame using efficient bulk insert
    df.to_sql(
        name=table_name,
        schema=schema,
        con=db_engine,
        if_exists="replace",   # replace table each run (append if desired)
        index=False,           # don't write the pandas index as a column
        method="multi",        # multi-row INSERT
        chunksize=1000
    )

    # Retrieve the DataFrame back from the database
    with db_engine.connect() as connection:
        result_df = pd.read_sql(
            text(f"SELECT * FROM {schema}.{table_name}"), connection
        )

    return result_df

def _parse_team_cell(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if pd.isna(v):
        return []
    s = str(v).strip()
    # Try JSON first: ["Aatrox","Ahri",...]
    try:
        out = json.loads(s)
        if isinstance(out, (list, tuple)):
            return list(out)
    except Exception:
        pass
    # Try Python literal: "['Aatrox','Ahri',...]"
    try:
        out = ast.literal_eval(s)
        if isinstance(out, (list, tuple)):
            return list(out)
    except Exception:
        pass
    # Fallback: strip brackets/braces and split
    s2 = s.strip('[](){}')
    return [p.strip().strip("'\"") for p in s2.split(',') if p.strip()]

def _normalize_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ('blue_team', 'red_team'):
        if col in df.columns:
            df[col] = df[col].apply(_parse_team_cell)
    return df

def fetch_from_sql(schema, table_name):
    """
    Fetches a DataFrame from a SQL database and normalizes team columns to lists.
    """
    db_engine = create_db_engine()
    with db_engine.connect() as connection:
        df = pd.read_sql(text(f'SELECT * FROM "{schema}"."{table_name}"'), connection)
    return _normalize_team_columns(df)

def get_pairs(row, team_col):
    """Generate all unique pairs of champions from a team."""
    return list(combinations(sorted(row[team_col]), 2))

def counter_normalization(a, b):
    """Calculate normalized counter delta using Elo-like formula."""
    if np.isnan(a) or np.isnan(b):  # Check if a or b is NaN
        return np.nan
    if a <= 0 or b <= 0 or a >= 100 or b >= 100:
        raise ValueError("Win rates must be between 0 and 100 (exclusive).")
    return 100 / (1 + 10 ** (np.log(100 / a - 1) - np.log(100 / b - 1)))

def synergy_normalization(a, b):
    """Calculate normalized synergy using Elo-like formula."""
    if np.isnan(a) or np.isnan(b):  # Check if a or b is NaN
        return np.nan
    if a <= 0 or b <= 0 or a >= 100 or b >= 100:
        raise ValueError("Win rates must be between 0 and 100 (exclusive).")
    return 100 / (1 + 10 ** (np.log(100 / a - 1) + np.log(100 / b - 1)))

def calculate_champion_winrates(df, min_games=30):
    """
    Calculate win rates for each champion based on the database.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing match data with 'blue_team', 'red_team', and 'target' columns.
        min_games (int): Minimum number of games required to calculate a reliable win rate.
    
    Returns:
        dict: A dictionary with champions as keys and their win rates as values. Champions with fewer than `min_games` will have NaN as their win rate.
    """
    from numpy import nan

    # Flatten the list of all champions and count their appearances
    all_champs = set(df['blue_team'].explode()).union(set(df['red_team'].explode()))
    champ_winrates = {}

    for champ in all_champs:
        # Mask for games where the champion appears
        mask = (df['blue_team'].apply(lambda team: champ in team)) | (df['red_team'].apply(lambda team: champ in team))
        total_games = mask.sum()
        
        if total_games >= min_games:  # Only calculate win rate if games exceed the threshold
            wins = (
                (df[mask]['blue_team'].apply(lambda team: champ in team) & (df[mask]['target'] == 1)).sum() +
                (df[mask]['red_team'].apply(lambda team: champ in team) & (df[mask]['target'] == 0)).sum()
            )
            champ_winrates[champ] = (wins / total_games) * 100  # Convert to percentage
        else:
            champ_winrates[champ] = nan  # Set to NaN for low-data champions

    return champ_winrates

def calculate_team_synergy(df, champ_winrates):
    def calculate_synergy(team):
        synergy_score = 0
        pairs = get_pairs({'team': team}, 'team')  # Generate all unique pairs
        for champ_a, champ_b in pairs:
            winrate_a = champ_winrates.get(champ_a, np.nan)
            winrate_b = champ_winrates.get(champ_b, np.nan)
            if np.isnan(winrate_a) or np.isnan(winrate_b):
                continue  # Ignore pairs with NaN win rates
            synergy_score += synergy_normalization(winrate_a, winrate_b)
        return float(synergy_score)

    blue_synergy = df['blue_team'].apply(calculate_synergy)
    red_synergy = df['red_team'].apply(calculate_synergy)

    return blue_synergy, red_synergy

def calculate_team_counters(df, champ_winrates):
    
    def calculate_counters(team, enemy_team):
        counter_score = 0.0
        for champ_a in team:
            for champ_b in enemy_team:
                wa = champ_winrates.get(champ_a, np.nan)
                wb = champ_winrates.get(champ_b, np.nan)
                if np.isnan(wa) or np.isnan(wb):
                    continue
                counter_score += float(counter_normalization(wa, wb))
        return float(counter_score)

    # Build 1-D Series explicitly
    blue_counters = pd.Series(
        (calculate_counters(bt, rt) for bt, rt in zip(df['blue_team'], df['red_team'])),
        index=df.index,
        dtype='float64'
    )
    red_counters = pd.Series(
        (calculate_counters(rt, bt) for bt, rt in zip(df['blue_team'], df['red_team'])),
        index=df.index,
        dtype='float64'
    )

    counter_delta = blue_counters - red_counters
    return counter_delta

def add_synergy_and_counter_scores(df, champ_winrates):
    """
    Add synergy and counter scores to the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing match data with 'blue_team' and 'red_team' columns.
        champ_winrates (dict): A dictionary with champions as keys and their win rates as values.

    Returns:
        pd.DataFrame: The DataFrame with added columns for synergy and counter scores.
    """
    # Calculate synergy scores
    df['synergy_blue'], df['synergy_red'] = calculate_team_synergy(df, champ_winrates)

    # Calculate counter delta (single column)
    df['counter_delta'] = calculate_team_counters(df, champ_winrates)

    return df