import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import io

def fetch_stock_data(ticker, strategy):
    stock = yf.Ticker(ticker)
    if strategy == "Intraday":
        data = stock.history(period="1d", interval="5m")
    elif strategy == "Swing":
        data = stock.history(period="1y")
    else:
        raise ValueError("Unknown strategy. Choose either 'Intraday' or 'Swing'.")
    
    if data.empty:
        raise ValueError("No data found for the specified ticker and strategy.")

    data['Price Change'] = data['Close'].diff()
    data['Direction'] = data['Price Change'].apply(lambda x: 1 if x > 0 else 0)
    data.dropna(inplace=True)
    return data

def fetch_financial_news(ticker):
    news = [
        {"title": f"Stock of {ticker} Shows Promising Growth", "link": f"https://news.com/{ticker}_growth"},
        {"title": f"{ticker} Faces Market Challenges", "link": f"https://news.com/{ticker}_challenges"},
        {"title": f"{ticker} Innovates with New Products", "link": f"https://news.com/{ticker}_innovations"},
        {"title": f"How {ticker} is Expanding Globally", "link": f"https://news.com/{ticker}_global_expansion"}
    ]
    return news

def analyze_sentiment_vader(news_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(news_text)
    return sentiment['compound']

def train_model(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Direction']
    if len(features) < 2 or len(target) < 2:
        raise ValueError("Insufficient data for training the model.")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, scaler, accuracy

def predict_stock_direction(model, scaler, current_data):
    features = current_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    if len(features) == 0:
        raise ValueError("No data available for prediction.")
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled[-1].reshape(1, -1))
    return "Upward" if prediction[0] == 1 else "Downward"

def identify_support_resistance(data):
    data['Support'] = data['Low'].rolling(window=14).min()
    data['Resistance'] = data['High'].rolling(window=14).max()
    return data

def visualize_candlestick_chart(data, ticker, strategy):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    )])

    if 'Support' in data.columns and 'Resistance' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Support'],
            line=dict(color='orange', width=1),
            mode='lines',
            name='Support'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Resistance'],
            line=dict(color='purple', width=1),
            mode='lines',
            name='Resistance'
        ))

    fig.update_layout(
        title=f"{strategy} Candlestick Chart with Support/Resistance for {ticker}",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    return fig

def make_decision(sentiment_score, predicted_trend):
    if sentiment_score > 0.2 and predicted_trend == "Upward":
        return "Buy"
    elif sentiment_score < -0.2 and predicted_trend == "Downward":
        return "Sell"
    else:
        return "Hold"

def generate_summary(ticker, sentiment_score, trend, decision, support, resistance, stock_data):
    summary = f"""
    Stock Summary for {ticker}
    ---------------------------
    Historical Data (Last 5 Rows):
    {stock_data.tail(5).to_string(index=True)}
    
    Sentiment Score: {sentiment_score:.2f}
    Predicted Trend: {trend}
    Recommendation: {decision}
    Support Level: {support}
    Resistance Level: {resistance}
    """
    return summary

def main():
    st.set_page_config(layout="wide")
    st.title("AI Financial Advisor with Sentiment Analysis and Candlestick Charts")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA): ").upper()

    if ticker:
        try:
            strategy = st.selectbox("Select Trading Strategy", ["Intraday", "Swing"])

            # Fetch stock data based on strategy
            stock_data = fetch_stock_data(ticker, strategy)

            col1, col2 = st.columns([4, 6])

            with col1:
                st.subheader(f"Historical Data for {ticker}")
                st.dataframe(stock_data.tail(10))

            with col2:
                st.subheader("Candlestick Chart")
                stock_data = identify_support_resistance(stock_data)
                fig = visualize_candlestick_chart(stock_data, ticker, strategy)
                st.plotly_chart(fig, use_container_width=True)

            st.write("Fetching financial news...")
            news = fetch_financial_news(ticker)
            sentiment_score = analyze_sentiment_vader(news[0]['title'])
            st.write(f"Sentiment Score for {ticker}: {sentiment_score:.2f}")

            sentiment_label = "Neutral"
            sentiment_color = "gray"
            if sentiment_score > 0.2:
                sentiment_label = "Positive"
                sentiment_color = "green"
            elif sentiment_score < -0.2:
                sentiment_label = "Negative"
                sentiment_color = "red"

            st.subheader("Sentiment Analysis")
            st.markdown(f"<p style='color:{sentiment_color}; font-size: 20px;'><strong>{sentiment_label}</strong></p>", unsafe_allow_html=True)
            st.progress(int((sentiment_score + 1) * 50))

            st.write("Training the model...")
            model, scaler, accuracy = train_model(stock_data)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            st.write("Predicting stock direction...")
            predicted_trend = predict_stock_direction(model, scaler, stock_data)
            st.write(f"Predicted Stock Trend: {predicted_trend}")

            decision = make_decision(sentiment_score, predicted_trend)
            st.write(f"Recommendation: {decision}")

            support = stock_data['Support'].iloc[-1]
            resistance = stock_data['Resistance'].iloc[-1]

            # Generate and provide the download option for the summary
            summary = generate_summary(ticker, sentiment_score, predicted_trend, decision, support, resistance, stock_data)
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"{ticker}_summary.txt",
                mime="text/plain"
            )

        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()