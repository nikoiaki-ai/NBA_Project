from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd



game_logs = LeagueGameLog(
    season="2025-26",
    season_type_all_star="Regular Season",
    player_or_team_abbreviation="P"
)

df = game_logs.get_data_frames()[0]

#Test to see that the api request works and we have a dataframe
print(df.shape)
print(df.columns.tolist())
print(df.head())

#Check to see if there are any duplicates
print(df.duplicated(subset=["PLAYER_ID", "GAME_ID"]).sum())

#Check to see if there are nulls
print(df.isnull().sum())

#Check to make sure we have the right season (25/26)
print(df["GAME_DATE"].min())
print(df["GAME_DATE"].max())

#Check and see if high counts arer due to people missing or not taking shots
print(df.loc[df["FG_PCT"].isna(), "FGA"].value_counts())
print(df.loc[df["FG3_PCT"].isna(), "FG3A"].value_counts())
print(df.loc[df["FT_PCT"].isna(), "FTA"].value_counts())

df.to_parquet(path="data/raw/player_game_logs_2025_26.parquet", engine='pyarrow', compression='snappy', index='false')
df_test = pd.read_parquet()