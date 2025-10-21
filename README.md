This project predicts the outcome of football matches (Win / Draw / Loss) using machine learning built from live match data scraped directly from Understat.com.

Project Overview:
Instead of relying on static Kaggle/HuggingFace datasets this project automatically scrapes match statistics such as goals, xG (expected goals), possession, shots, and passing data and many more from Understat. The scraped data is cleaned, preprocessed, and converted into a machine learning–ready format. Models are then trained to predict match outcomes based on both team and opponent performance metrics.

Workflow:

(1). Web Scraping:
- Uses requests and BeautifulSoup to extract data for top leagues (e.g., EPL).
- Parses match IDs, team names, and key stats (xG, xGA, goals, xPTS, possession, etc.).
- Stores the data into a structured DataFrame and exports it to a CSV file.

(2). Data Cleaning & Preprocessing:
- Converts string-based stats (e.g., percentages, “xG per 90”) to numeric form.
- Handles missing values and normalizes key performance indicators.
- Prepares separate or combined home/away records for model training.

(3). Feature Engineering:
- Builds custom features such as:
- Recent team form (last 5 games)
- Opponent strength
- Venue advantage
- Rolling averages of team stats (xG, goals, possession)
- Is it a derby?

(4). Model Training (In Progress):
- Trains a Random Forest Classifier as the baseline.
- Evaluates model performance using accuracy, precision, recall, and F1-score.
- Tests alternative models (e.g., Logistic Regression, XGBoost) for optimization.

(5). Tech Stack:
- Python
- Requests, BeautifulSoup4 – Web scraping
- Pandas, NumPy – Data processing and feature engineering
- Scikit-learn – Model building and evaluation

(6). Future Plans for now:
- Adding more models and making them work together.
- Build a small dashboard to visualize predictions interactively, (Most probably StreamLit)
