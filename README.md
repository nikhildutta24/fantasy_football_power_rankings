# Fantasy Football Power Rankings
A Python-based analytics tool that generates data-driven power rankings for ESPN Fantasy Football leagues. The model combines recent scoring trends, strength of schedule, injury impact, and a walk-forward predictive model to produce both static and dynamically weighted power scores.

---

## Features
- **Automated data fetching** from ESPN's private league API
- **Injury-aware rankings** with position-weighted adn starter/bench-aware impact scoring
- **Walk-forward predictive model** using linear regression trained on past weeks to avoid lookahead bias
- **Two Ranking Systems**: a hand-tuned static power scoreand a data-learned dynamic power score
  - (Further fine-tuning in progress to optimize data-learned model and ultimately phase out the hand-tuned model)
- **Luck metric** comparing actual wins to expected wins based on league-wide scoring each week

---

## Project Structure
```
├── data/
│   └── raw/
│       ├── weekly_stats.csv       # Per-team, per-week results
│       ├── fantasy_roster.csv     # Current rosters with slot positions
│       └── weekly_injuries.csv    # Scraped NFL injury report
├── src/
│   └── fetch_data.py              # Pulls league data from ESPN API
│   └── fetch_injuries.py          # Scrapes injury report from ESPN
│   └── power_rankings.py          # Computes and outputs power rankings
├── .env                           # Local credentials (not committed)
└── README.md
```
---

## Setup
### 1. Clone the repository
```bash
git clone https://github.comyourusername/fantasy-power-rankings.git
cd fantasy-power-rankings
```

### 2. Installdependencies
```bash
pip install espn_api pandas scikit-learn matplotlib beautifulsoup4 requests python-dotenv
```

### 3. Configure your ".env" file

Create a ".env" file in the root directory:
```
LEAGUE_ID=your_league_id
YEAR=2024
ESPN_S2=your_espn_s2_cookie
ESPN_SWID=your_swid_cookie
```
To find your `ESPN_S2` and `SWID` cookies, log into ESPN Fantasy, open your browser's developer tools, go to **Application → Cookies → espn.com**, and copy the values for `espn_s2` and `SWID`.

> ⚠️ Never commit your `.env` file. Add it to `.gitignore`.
---

## Usage
Run the scripts in order each week:
```bash
# 1. Fetch league scores and rosters
python fetch_data.py

# 2. Scrape current injury report
python fetch_injuries.py
(May require re-running closer to the start of the week to account for new downgrade and upgrade designations

# 3. Generate power rankings
python power_rankings.py
```

---

## How It Works

### Static Power Score

A weighted combination of z-scored features:

| Feature | Weight | Description |
|---|---|---|
| Recent Scoring (3-week rolling avg) | +0.30 | How hot a team is right now |
| Season Average | +0.25 | Overall scoring baseline |
| Avg Point Differential | +0.20 | Margin of victory/defeat |
| Strength of Schedule | +0.15 | Quality of opponents faced |
| Last Week Score | +0.05 | Most recent performance |
| Luck | +0.05 | Actual wins minus expected wins |
| Consistency (std dev penalty) | -0.10 | Penalizes high variance scoring |
| Injury Impact | -0.10 | Penalizes teams with injured players |

### Dynamic Power Score

Weights are learned automatically via linear regression trained on walk-forward out-of-sample predictions, so the model reflects
what features actually predict future scoring rather than relying on hand-tuned and manually decided values.

### Injury Impact Scoring

Each injured player contributes a score based on their status, position, and whether they are a starter:

- **Status**: IR = 1.0, Doubtful = 0.5, Questionable/Probable = 0.25
- **Position multiplier**: QB = 3.0x, RB = 2.5x, WR = 2.0x, TE = 1.5x, K = 1.0x
- **Starter multiplier**: Starters = 1.0x, Bench = 0.25x

---

## Requirements

- Python 3.8+
- A private or public ESPN Fantasy Footbal League
- ESPN account credentials (private leagues only)

---

## Future Enhancements

- Weekly matchup predictions with win probabilities
- Roster optimization suggestions (start/sit recommendations)
- Player-level projections integration
- Automated weekly report generation
- Scoring data visualizations
