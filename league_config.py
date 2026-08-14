"""
Central league configuration. Add/remove leagues here — nothing else
needs touching to bring a new one online once its data checks out.

has_odds=False leagues (League One/Two right now) run forecast-only:
ratings and match probabilities, no edge/traffic-light/CLV layer,
because there's no market price to compare against.
"""

SEASON = "2526"  # last completed season — used for initial model fitting
DATA_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "Premier League": {
        "data_code": "E0",
        "odds_key": "soccer_epl",
        "has_odds": True,
    },
    "EFL Championship": {
        "data_code": "E1",
        "odds_key": "soccer_efl_champ",
        "has_odds": True,
    },
    "League One": {
        "data_code": "E2",
        "odds_key": None,
        "has_odds": False,
    },
    "League Two": {
        "data_code": "E3",
        "odds_key": None,
        "has_odds": False,
    },
    "Bundesliga": {
        "data_code": "D1",
        "odds_key": "soccer_germany_bundesliga",
        "has_odds": True,
    },
    "Serie A": {
        "data_code": "I1",
        "odds_key": "soccer_italy_serie_a",
        "has_odds": True,
    },
    "La Liga": {
        "data_code": "SP1",
        "odds_key": "soccer_spain_la_liga",
        "has_odds": True,
    },
    "Ligue 1": {
        "data_code": "F1",
        "odds_key": "soccer_france_ligue_one",
        "has_odds": True,
    },
}

# Odds API name -> football-data.co.uk name, per league.
# Only English football is filled in from what we've already confirmed
# working. The others are STUBS — run_all_leagues.py will print any
# fixture it can't match so you can top these up from real output,
# same as we did for the Premier League.
TEAM_NAME_MAPS = {
    "Premier League": {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Tottenham Hotspur": "Tottenham",
        "Wolverhampton Wanderers": "Wolves",
        "Brighton and Hove Albion": "Brighton",
        "West Ham United": "West Ham",
        "Leeds United": "Leeds",
        "Ipswich Town": "Ipswich",
        "Hull City": "Hull",
        "Coventry City": "Coventry",
    },
    "EFL Championship": {},
    "League One": {},
    "League Two": {},
    "Bundesliga": {},
    "Serie A": {},
    "La Liga": {},
    "Ligue 1": {},
}


def data_url(league_name: str) -> str:
    code = LEAGUES[league_name]["data_code"]
    return DATA_URL_TEMPLATE.format(season=SEASON, code=code)


def normalise_name(league_name: str, odds_api_name: str) -> str:
    return TEAM_NAME_MAPS.get(league_name, {}).get(odds_api_name, odds_api_name)
