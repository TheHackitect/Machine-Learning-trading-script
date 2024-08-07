import sys
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QGridLayout,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QIcon
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

class TradingApp(QMainWindow):
    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
        self.interval = '1m'  # Default interval
        self.capital = 100.0  # Default test capital
        
        # Load model with specified loss function
        self.model = load_model(f'{ticker}_lstm_model.h5', compile=False)
        self.model.compile(optimizer='adam', loss=MeanSquaredError())

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.initUI()
        self.update_data()

    def initUI(self):
        self.setWindowTitle('Trading Signals')
        self.setGeometry(100, 100, 1200, 800)
        
        layout = QVBoxLayout()
        
        # Create settings panel
        self.settings_panel = QWidget()
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel('Ticker:'), 0, 0)
        self.ticker_input = QLineEdit(self.ticker)
        settings_layout.addWidget(self.ticker_input, 0, 1)

        settings_layout.addWidget(QLabel('Update Interval (minutes):'), 1, 0)
        self.update_interval_input = QSpinBox()
        self.update_interval_input.setValue(1)
        settings_layout.addWidget(self.update_interval_input, 1, 1)
        
        settings_layout.addWidget(QLabel('Test Capital ($):'), 2, 0)
        self.capital_input = QDoubleSpinBox()
        self.capital_input.setValue(self.capital)
        settings_layout.addWidget(self.capital_input, 2, 1)

        self.update_button = QPushButton('Update Settings')
        self.update_button.clicked.connect(self.update_settings)
        settings_layout.addWidget(self.update_button, 3, 0, 1, 2)
        
        self.settings_panel.setLayout(settings_layout)
        layout.addWidget(self.settings_panel)
        
        # Create signal display panel
        self.signal_panel = QWidget()
        signal_layout = QVBoxLayout()
        
        self.signal_label = QLabel('Fetching data...')
        self.signal_label.setAlignment(Qt.AlignCenter)
        signal_layout.addWidget(self.signal_label)
        
        self.signal_panel.setLayout(signal_layout)
        layout.addWidget(self.signal_panel)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def update_settings(self):
        self.ticker = self.ticker_input.text()
        self.interval = f'{self.update_interval_input.value()}m'
        self.capital = self.capital_input.value()
        self.update_data()

    def update_data(self):
        df = yf.download(self.ticker, period='1d', interval=self.interval)
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        X = []
        n_lags = 60
        for i in range(n_lags, len(scaled_data)):
            X.append(scaled_data[i - n_lags:i, 0])
        
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        
        latest_prediction = predictions[-1, 0]
        latest_close = df['Close'].values[-1]
        
        if latest_close < latest_prediction:
            signal = 'BUY'
            action = f'If you buy now with ${self.capital}, you might gain ${self.capital * (latest_prediction - latest_close) / latest_close:.2f} after 1 minute.'
        elif latest_close > latest_prediction:
            signal = 'SELL'
            action = f'If you sell now with ${self.capital}, you might avoid a loss of ${self.capital * (latest_close - latest_prediction) / latest_close:.2f} after 1 minute.'
        else:
            signal = 'HOLD'
            action = f'Hold your position. No significant change expected in the next minute.'

        self.signal_label.setText(f'Latest Signal: {signal}\nClose Price: {latest_close}\nPredicted Price: {latest_prediction}\n{action}')
        
        # Plot graph
        plt.figure(figsize=(10, 5))
        plt.plot(df.index[-len(predictions):], predictions, label='Predicted')
        plt.plot(df.index, df['Close'], label='Actual')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('price_plot.png')
        plt.close()

        # Update data periodically
        QTimer.singleShot(self.update_interval_input.value() * 60000, self.update_data)  # Update based on user-defined interval

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ticker = 'LTC-USD'  # Default ticker
    ex = TradingApp(ticker)
    ex.show()
    sys.exit(app.exec_())
