import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf

# Define the custom mse loss function
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the yfinance dataset
df = yf.download('LTC-USD', start='2018-01-01', end='2024-07-01')

# Ensure the index is a DatetimeIndex
df['Date'] = pd.to_datetime(df.index, errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df.set_index('Date', inplace=True)

# Feature engineering
df['MA50'] = df['Close'].rolling(window=50).mean()
df['Volatility'] = df['Close'].rolling(window=50).std()

# Drop any remaining NaNs after feature engineering
df.dropna(inplace=True)

# Scale the data for training
scaler_features = MinMaxScaler()
scaled_data = scaler_features.fit_transform(df[['Close', 'MA50', 'Volatility']])

# Scale the target feature separately for inverse transformation
scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(df[['Close']])

# Function to prepare data for LSTM model
def prepare_data(data, n_lags):
    X, y = [], []
    for i in range(len(data) - n_lags):
        X.append(data[i:i+n_lags])
        y.append(data[i+n_lags, 0])  # Only target feature
    return np.array(X), np.array(y)

# Function to train and save the model
def train_and_save_model(X_train, y_train, X_test, y_test, period):
    print(f"Training model for period: {period}")
    
    # Print the shape of data for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Check if data is sufficient
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Not enough data to train the model for period: {period}")
        return
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_lags, X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=custom_mse)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save(f'lstm_model_{period}.keras')
    
    # Plot training and validation loss
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Model Training and Validation Loss ({period})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to load the model and make predictions
def load_and_predict_model(X_test, period):
    # Load the trained LSTM model with custom objects
    custom_objects = {'custom_mse': custom_mse}
    loaded_model = load_model(f'lstm_model_{period}.keras', custom_objects=custom_objects)
    
    # Make predictions with the loaded model
    y_pred_loaded = loaded_model.predict(X_test)
    return y_pred_loaded

# Prepare and train models for different periods
periods = ['D', 'W', 'M', 'Y']
n_lags = 50
results = {}

for period in periods:
    # Resample data
    resampled_df = df.resample(period).last()
    resampled_df.dropna(inplace=True)
    
    if len(resampled_df) < n_lags:
        print(f"Not enough data after resampling for period: {period}")
        continue
    
    # Scale data
    resampled_scaled_data = scaler_features.transform(resampled_df[['Close', 'MA50', 'Volatility']])
    
    # Scale target
    resampled_scaled_target = scaler_target.transform(resampled_df[['Close']])
    
    # Prepare data
    X, y = prepare_data(resampled_scaled_data, n_lags)
    
    # Split data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train and save the model
    train_and_save_model(X_train, y_train, X_test, y_test, period)
    
    # Load and predict
    y_pred_loaded = load_and_predict_model(X_test, period)
    
    # Inverse transform the predictions and actual values
    y_test_reshaped = y_test.reshape(-1, 1)
    y_pred_reshaped_loaded = y_pred_loaded.reshape(-1, 1)
    
    y_test_inverse = scaler_target.inverse_transform(y_test_reshaped)
    y_pred_inverse_loaded = scaler_target.inverse_transform(y_pred_reshaped_loaded)
    
    # Reshape back to original shape
    y_test_inverse = y_test_inverse.flatten()
    y_pred_inverse_loaded = y_pred_inverse_loaded.flatten()
    
    # Store results
    results[period] = {
        'y_test_inverse': y_test_inverse,
        'y_pred_inverse_loaded': y_pred_inverse_loaded
    }

# Output and plot results
for period in periods:
    if period not in results:
        continue
    print(f"Results for {period}:")
    current_price = df['Close'].iloc[-1]
    predicted_price = results[period]['y_pred_inverse_loaded'][-1]
    
    # Determine buy/sell signal
    if predicted_price > current_price:
        signal = 'Buy'
        expected_profit = predicted_price - current_price
    else:
        signal = 'Sell'
        expected_profit = current_price - predicted_price

    # Risk management
    stop_loss = current_price * 0.95  # 5% below the current price
    take_profit = current_price * 1.05  # 5% above the current price

    # Output Suggested Trades
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"Signal: {signal}")
    print(f"Expected Profit: ${expected_profit:.2f}")
    print(f"Stop-Loss Level: ${stop_loss:.2f}")
    print(f"Take-Profit Level: ${take_profit:.2f}")
    
    # Plot the predictions vs actual values
    plt.figure(figsize=(14, 7))
    plt.plot(results[period]['y_test_inverse'], label='Actual Price')
    plt.plot(results[period]['y_pred_inverse_loaded'], label='Predicted Price')
    plt.title(f'LSTM Model Predictions vs Actual Prices ({period})')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
