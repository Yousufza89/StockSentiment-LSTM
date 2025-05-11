# StockSentiment-LSTM
A deep learning-based project for predicting hourly stock prices by incorporating real-time sentiment analysis from Twitter and news sources using Stacked LSTM and GRU models. The project aims to demonstrate the impact of public sentiment on short-term price movements.

# 📈 Stock Price Prediction Using Sentiment Analysis

A deep learning project designed to predict **hourly stock prices** using both **historical price data** and **sentiment analysis** from **Twitter and online news**. The model employs advanced LSTM architectures to understand temporal patterns and the influence of public mood.

---

## 👨‍💻 Team Members

- **Muhammad Yousuf Rehan** – 22K-4457 *(Group Leader)*
- **Murtaza Johar** – 22K-4508 


---

## 🎯 Project Objective

To develop a stock price prediction model that integrates real-time sentiment data extracted from Twitter and news articles, improving the accuracy of traditional time-series prediction models by leveraging NLP-driven sentiment features.

---

## 🗂️ Dataset Sources

- **Stock Price Data**: [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- **Twitter Data**: Collected using `Tweepy` with keyword filters like `#TSLA`, `#Tesla`, `Elon Musk`
- **News Data**: Fetched using [NewsAPI.org](https://newsapi.org/) and parsed for sentiment

---

## 📊 Features

- **Price Features**: Open, High, Low, Close, Volume
- **Sentiment Features**: Compound, Positive, Negative, Neutral
- **Hourly Aggregation** for sentiment scores
- **Multivariate Time Series Prediction**

---

## 🧠 Models Used

- Baseline: Linear Regression
- Main: **Stacked LSTM**
- Alternate: **Bidirectional GRU**

---

## 📈 Evaluation Metrics

| Model           | MAE   | RMSE  | R² Score |
|----------------|--------|--------|----------|
| Linear Regress | 4.76  | 6.20  | 0.51     |
| Stacked LSTM   | 2.85  | 3.92  | 0.79     |
| Bi-GRU         | 3.02  | 4.10  | 0.76     |

---

## 📁 Folder Structure

StockSentiment-LSTM/
│
├── 📁 data/
│ ├── stock_data.csv
│ ├── twitter_data.csv
│ └── news_data.csv
│
├── 📁 models/
│ ├── lstm_model.h5
│ └── gru_model.h5
│
├── 📁 notebooks/
│ └── EDA_and_Training.ipynb
│
├── 📁 utils/
│ ├── sentiment_analysis.py
│ └── data_preprocessing.py
│
├── training.py
├── requirements.txt
└── README.md


---

## 🛠️ Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/StockSentiment-LSTM.git
   cd StockSentiment-LSTM
   
# Create virtual environment (optional but recommended)  
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Run training:
python training.py

## ✅ Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow
nltk
vaderSentiment
textblob
tweepy
yfinance
newsapi-python

## 📌 Future Improvements
Real-time data streaming for live predictions

Integration of BERT for more accurate sentiment scoring

Cross-stock and sectoral analysis

Deployment via Flask/Streamlit app

## 🤝 Acknowledgements
Yahoo Finance

Twitter Developer API

NewsAPI.org

Sentiment Tools: VADER, TextBlob
