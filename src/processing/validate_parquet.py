
#%%
import pandas as pd
from pathlib import Path
df__loaded = pd.read_parquet(
    "data/raw/player_game_logs_2025_26.parquet"
)
print(df__loaded.head(5))
# 
# %%
import os 
print("Current Directory: ", os.getcwd())
print("File exists:", Path("data/raw/player_game_logs_2025_26.parquet").exists())