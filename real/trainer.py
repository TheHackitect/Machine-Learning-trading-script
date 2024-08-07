import argparse
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import datetime

# Function to download and preprocess data
def download_data(ticker):
    df = yf.download(ticker, start='2010-01-01')
    return df

# Function to create dataset for LSTM
def create_dataset(df, n_lags=60):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(n_lags, len(scaled_data)):
        X.append(scaled_data[i - n_lags:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to build and train LSTM model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return model

# Function to predict and generate trading signals
def generate_signals(model, X, scaler, df):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    df['Predicted_Close'] = np.nan
    df['Predicted_Close'].iloc[-len(predictions):] = predictions.flatten()
    
    df['Signal'] = 0
    df['Signal'][df['Close'] < df['Predicted_Close']] = 1
    df['Signal'][df['Close'] > df['Predicted_Close']] = -1
    return df

# Function to visualize the results
def visualize_results(df, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Actual Close Price')
    plt.plot(df['Predicted_Close'], label='Predicted Close Price')
    plt.title(f'{ticker} Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Main function
def main(ticker):
    df = download_data(ticker)
    
    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]
    
    X_train, y_train, scaler_train = create_dataset(train_df)
    X_test, y_test, scaler_test = create_dataset(test_df)
    
    model = build_and_train_model(X_train, y_train, X_test, y_test)
    signals_df = generate_signals(model, np.vstack((X_train, X_test)), scaler_test, df)
    
    visualize_results(signals_df, ticker)
    
    # Save the model
    model.save(f'{ticker}_lstm_model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an LSTM model for trading signals.')
    parser.add_argument('ticker', type=str, help='Ticker symbol for the asset.')
    
    args = parser.parse_args()
    main(args.ticker)
