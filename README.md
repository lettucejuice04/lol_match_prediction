# predicting_lol_c1_matches

League of Legends Draft Outcome Prediction
This project predicts which team will win a League of Legends match using only the champion draft. It pulls fresh game data from Riot’s API, engineers draft-based features like champion presence, synergy, and counters, and trains gradient-boosted tree models to classify the winner.

What I built
End-to-end pipeline: data acquisition -> preprocessing -> feature engineering -> modeling -> evaluation
SQL-backed storage for consistent, fast iteration
A model that can score arbitrary champion drafts (e.g., current or hypothetical comps)
Data collection
Source: Riot API (NA, EUW, KR)
Scope: Challenger‑tier players, recent ranked solo matches
Process:
Get Challenger ladder, fetch player PUUIDs per region
Pull recent match IDs and match details
Map champion IDs to names via Data Dragon
Extract for each match: blue team champs, red team champs, and the winning side
Data is saved to PostgreSQL via SQLAlchemy for reliability and speed, then reloaded for preprocessing
Preprocessing and data cleaning
Normalize team columns into clean Python lists
Remove problematic rows:
Missing teams
Teams not exactly 5 champions
Duplicate champions on the same team
Unknown/out‑of‑list champions (checked against a canonical champion list)
Create target variable: 1 if blue side wins, 0 otherwise
Feature engineering
To represent drafts numerically for ML, I engineered three families of features:

Champion presence (side-agnostic)
For every champion in the global champion list, a binary feature indicates whether they appear in the match (regardless of team).
Team synergy
Pairwise ally‑ally interactions within a team.
Aggregated with an Elo‑style normalization using champion win rates derived from the dataset.
Team counters
Cross‑team interactions comparing each champion vs all opponents.
Also normalized with an Elo‑style function and aggregated into a “counter delta” signal.
The final feature matrix combines:

All champion presence indicators
Synergy scores for blue and red teams
A counter delta score capturing draft advantage between teams
Modeling and evaluation
Models tried: LightGBM and XGBoost
Data split: train/test with scikit‑learn
Selected model: XGBoost (comparable to LightGBM; chosen for final runs)
Hyperparameter search: RandomizedSearchCV for accuracy
Observed performance: around 76% test accuracy on this dataset, using draft features alone
Why gradient boosting?

Handles high‑dimensional, sparse binary features well
Captures non‑linear interactions between champions
Robust to noisy frequency/winrate signals
Insights
Feature importance suggests the model emphasizes champions and interactions that align with high‑elo meta expectations.
Lower‑impact champions (in this dataset) tend to be those less prevalent or less consistently impactful at high tiers.
Real-time and hypothetical draft scoring
The trained model can evaluate any proposed draft:
Provide blue and red champion lists
Recompute presence/synergy/counter features
Predict probability of a blue‑side win
This enables quick “what‑if” analysis for different compositions.
Reproducibility
Notebooks:
01_data_acquisition.ipynb: Pull Riot data and store in SQL
02_data_preprocessing.ipynb: Clean data, engineer features, save cleaned_features.pkl
03_data_modeling.ipynb: Build matrices, train models, tune hyperparameters, run mock predictions
Data artifact: cleaned_features.pkl for fast reload in modeling
Tech stack
Python: pandas, scikit‑learn, XGBoost, LightGBM
Data: requests, dotenv, Data Dragon
Storage: PostgreSQL + SQLAlchemy (psycopg2)
Supporting libs: itertools, numpy
Limitations and next steps
Player skill, role swaps, patch changes, and off‑meta picks aren’t modeled directly.
Winrate baselines are derived from the collected set; performance can drift across patches/regions.
Future improvements:
Role- and lane‑aware features
Patch/version conditioning
Ban information and pick order dynamics
Calibrated probabilities and reliability diagrams
Periodic retraining with fresh data

I built a full pipeline that uses Riot API data to learn draft-only signals for match outcomes. With champion presence, synergy, and counter features feeding a boosted-tree model, it achieves ~76% test accuracy and can score any draft in real time.
