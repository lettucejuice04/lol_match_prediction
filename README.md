# Predicting LoL Match Outcomes (Draft Only)

This project predicts which team will win a League of Legends match using only the champion draft. It pulls high-elo game data from the Riot API, engineers draft-based features like champion presence, synergy, and counters, and trains **gradient-boosted tree models** to classify the winner.

---

## Project Highlights

* **Goal:** Predict match outcome using **only champion draft** (10 champions).
* **Data Scope:** Riot API (NA, EUW, KR), **Challenger-tier** ranked solo matches.
* **Storage:** **PostgreSQL** via SQLAlchemy for reliable, fast iteration.
* **Model:** **XGBoost** (Gradient-Boosted Tree).
* **Performance:** Achieved **~76% test accuracy**.
* **Functionality:** Can score any current or hypothetical draft in real-time.

---

## Feature Engineering Summary

The final feature matrix combines three types of signals:

* **Champion Presence:** Binary indicators for every champion in the match.
* **Team Synergy:** Pairwise ally-ally interactions, normalized using an Elo-style function.
* **Team Counters:** Cross-team interactions, aggregated into a "counter delta" score, also Elo-normalized.

---

## Pipeline Structure

1.  `01_data_acquisition.ipynb`: Pulls Riot data and stores in SQL.
2.  `02_data_preprocessing.ipynb`: Cleans data, engineers features, saves `cleaned_features.pkl`.
3.  `03_data_modeling.ipynb`: Trains models, tunes hyperparameters, and runs predictions.

---

## Limitations

* Player skill, role swaps, and off-meta picks are **not** modeled directly.
* Performance can drift across patches due to reliance on win rate baselines derived from the collected dataset.
