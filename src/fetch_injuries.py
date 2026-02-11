import requests
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
INJURY_CSV_PATH = BASE_DIR / "data" / "raw" / "weekly_injuries.csv"


url = "https://www.espn.com/nfl/injuries"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/144.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

tables = soup.find_all("table", {"class": "Table"})

all_injuries = []
fantasy_positions = {"QB", "RB", "WR", "TE", "PK"}

for table in tables:
    rows = table.find_all("tr")
    for row in rows[1:]:
        cols = row.find_all("td")
        if not cols:
            continue
        name_col = cols[0].find("a")
        name = name_col.get_text(strip=True) if name_col else cols[0].get_text(strip=True)
        pos = cols[1].get_text(strip=True)
        est_return = cols[2].get_text(strip=True)
        status = cols[3].get_text(strip=True)
        comment = cols[4].get_text(strip=True) if len(cols) > 4 else ""
        
        all_injuries.append([name, pos, est_return, status, comment])

injuries_df = pd.DataFrame(all_injuries, columns=["NAME", "POS", "EST_RETURN_DATE", "STATUS", "COMMENT"])
injuries_df = injuries_df[injuries_df["POS"].isin(fantasy_positions)]
relevant_statuses = {"Questionable", "Probable", "Doubtful", "Active", "Injured Reserve"}
injuries_df = injuries_df[injuries_df["STATUS"].isin(relevant_statuses)]

injuries_df["POS"] = injuries_df["POS"].replace({
    "PK": "K"
}) 
        

print(injuries_df.head())
print(f"Total injured fantasy players: {len(injuries_df)}")

INJURY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
injuries_df.to_csv(INJURY_CSV_PATH, index=False)

