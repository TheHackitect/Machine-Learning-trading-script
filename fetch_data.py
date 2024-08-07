import sys
import argparse
import os
import cryptocompare
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class HistoricalDataFetcher:
    def __init__(self, source, start_date, end_date):
        self.source = source
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.data_frames = []

    def fetch_data(self):
        if self.source == 'cryptocompare':
            self._fetch_from_cryptocompare()
        elif self.source == 'yfinance':
            self._fetch_from_yfinance()
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def _fetch_from_cryptocompare(self):
        start_date = self.start_date
        while start_date < self.end_date:
            to_date = start_date + timedelta(days=2000)
            if to_date > self.end_date:
                to_date = self.end_date

            print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")

            historical_data = cryptocompare.get_historical_price_day('LTC', currency='USD', limit=2000, toTs=int(to_date.timestamp()))

            if historical_data:
                df = pd.DataFrame(historical_data)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)

                self.data_frames.append(df)

            start_date = to_date + timedelta(days=1)

    def _fetch_from_yfinance(self):
        current_start_date = self.start_date
        while current_start_date < self.end_date:
            current_end_date = current_start_date + timedelta(days=365)
            if current_end_date > self.end_date:
                current_end_date = self.end_date

            print(f"Fetching data from {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

            ltc_data_chunk = yf.download('LTC-USD', start=current_start_date.strftime('%Y-%m-%d'), end=current_end_date.strftime('%Y-%m-%d'))

            if not ltc_data_chunk.empty:
                self.data_frames.append(ltc_data_chunk)

            current_start_date = current_end_date + timedelta(days=1)

    def save_to_csv(self, filename):
        if not os.path.exists('data'):
            os.makedirs('data')
        full_data = pd.concat(self.data_frames)
        file_path = os.path.join('data', filename)
        full_data.to_csv(file_path)
        print(f"Data successfully saved to {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Fetch historical data for Litecoin.')
    parser.add_argument('--source', type=str, required=True, choices=['cryptocompare', 'yfinance'], help='Data source to fetch from.')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD).')
    args = parser.parse_args()

    fetcher = HistoricalDataFetcher(args.source, args.start, args.end)
    fetcher.fetch_data()
    output_filename = f"{args.source}-data.csv"
    fetcher.save_to_csv(output_filename)

if __name__ == '__main__':
    main()
