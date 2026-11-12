# Health Care Chatbot

> **Attribution:** This repository contains **Health-Care-Chatbot**, created by **Amber Kakkar** and released under the MIT License (see [`Health-Care-Chatbot-main/LICENSE`](Health-Care-Chatbot-main/LICENSE)). It is kept here for study. Original authorship belongs to Amber Kakkar.

**Not medical advice.** The bot answers from a fixed `intents.json` and is for learning only.

## What's inside

| File | Purpose |
|---|---|
| `training_py.py` | Tokenises `intents.json` and trains a small Keras dense network → `chatbotmodel.h5` |
| `chatbot_py.py` | Loads the model, `words.pkl` and `classes.pkl` and predicts an intent |
| `gui.py` | Tkinter chat window |
| `Data_Analysis.ipynb` | Exploratory analysis of `heart-disease.csv` |

## Run

```bash
cd Health-Care-Chatbot-main
pip install tensorflow nltk numpy
python training_py.py   # optional: retrain
python gui.py
```
