import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

org_features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote']
features=['pre_open_price', 'pre_high_price', 'pre_low_price', 'pre_close_price', 'pre_volume', 'pre_quote_asset_volume', 'pre_number_of_trades', 'pre_taker_buy_base', 'pre_taker_buy_quote']

# Initialize model, criterion, and optimizer
input_dim = len(features)
hidden_dim = 64
n_heads = 4
n_layers = 2
independent_var = 'pivot_price'
num_epochs      = 10
# Sequence length (number of days)
#sequence_length = 42 # 6x7
sequence_length = 6*3

# Dataset for rolling window, continuously train the past 7 days prices and use the model to predict the 8 day's price.
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        # data contains features + independent_var
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]   # Features (exclude independent_var)
        y = self.data[idx, -1]    # Independent var
        x = x.astype(np.float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for _ in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            #print(f"[DEBUG]x_batch={x_batch}")
            #print(f"[DEBUG]y_pred={y_pred.squeeze()}")
            #print(f"[DEBUG]y_batch={y_batch}")
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    return model

# Function to perform rolling training and prediction
def rolling_train_and_predict(values, seq_len, model, criterion, optimizer, num_epochs):
    """
    values = [date + features + independent_var],
    e.g. [Timestamp(), 0.9510539770126344 1.3134870529174805 0.6941869854927063 0.8320050239562988 87364276.0 0.8636373281478882]
    """
    df_pred = pd.DataFrame(columns=['date', f'{independent_var}',f'pred_{independent_var}', 'loss'])
    for start_idx in range(len(values) - seq_len - 1):
        end_idx = start_idx + seq_len
        rolling_data = values[start_idx:end_idx, 1:]
        #print(f"[DEBUG]start_idx={start_idx};end_idx={end_idx}")
        #print(f"[DEBUG]dates={dates}")
        #print(f'[DEBUG]rolling_data={rolling_data}')

        dataset = TimeSeriesDataset(rolling_data)
        train_loader = DataLoader(dataset,
                                  batch_size=sequence_length,
                                  shuffle=False)

        # Train the model on the rolling window
        model = train_model(model, train_loader, criterion, optimizer, num_epochs)

        # Predict the next day's pivot price
        model.eval()
        with torch.no_grad():
            test_date = values[end_idx, 0]
            x = np.array(values[end_idx, 1:-1], dtype=np.float32)
            test_input = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            test_output = values[end_idx, -1]
            #print(f'[DEBUG]test_input={test_input}')
            #print(f'[DEBUG]test_output={test_output}')
            prediction = model(test_input).item()
            loss = np.abs(prediction - test_output)
            df_pred.loc[len(df_pred)] = [test_date, test_output, prediction, loss]
            if end_idx % 10 == 0:
              print(f'[DEBUG]start_idx={start_idx};end_idx={end_idx}')
              print(f'test_date={test_date};actual={test_output};prediction={prediction};loss={loss}')
    return df_pred

def min_max_dates(data):
    # Calculate days between min and max dates
    min_date = data['datetime'].min()
    max_date = data['datetime'].max()
    print(f'min_date={min_date}; max_date={max_date}')
    # days between dates
    days_between = (pd.to_datetime(max_date) - pd.to_datetime(min_date)).days
    print(f'days_between={days_between}')
    # change days to years, months and days
    years = days_between // 365
    months = (days_between % 365) // 30
    days = days_between % 30
    print(f'years={years}; months={months}; days={days}')
  
# Take raw Binance market price data and create a new dataframe
#  with all the previous prices as features, and the current pivot 
#  price as the target variable.
def generate_training_dataset(data):
    # Filter out rows with ignore=0
    data = data[data['ignore'] == 0]
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
    data['datetime'] = data['close_time']
    # Calculate pivot price
    #data['pivot_price'] = (data['high_#rice'] + data['low_price'] + data['close_price']) / 3
    data['pivot_price'] = data['close_price']

    for col in org_features:
        data[f'pre_{col}'] = data[col].shift(1)
    
    # reorganize columns
    new_df = data[['datetime'] + ['pivot_price'] + [col for col in data.columns if 'pre' in col]].copy()
    new_df.dropna(inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def plot_predictions_actuals(predictions):
  # Plot dates, predictions, actuals, losses
  plt.figure(figsize=(12, 6))

  plt.subplot(2, 1, 1)
  plt.plot(predictions['date'], predictions[f'pred_{independent_var}'], label='Predicted Price', color='coral')
  plt.plot(predictions['date'], predictions[f'{independent_var}'], color='gray', alpha=0.8, label='Actual Price')

  # plot actuals with shadow filled
  plt.fill_between(predictions['date'], predictions[f'{independent_var}'], color='darkgray', alpha=0.5)
  #disable xtick
  plt.xticks([])
  plt.xlabel('Date')
  #plt.xticks(rotation=90)
  plt.ylabel('Price')
  plt.title('Predicted vs Actual Price')
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.plot(predictions['date'], predictions['loss'], label='Loss', color='g')
  plt.xlabel('Date')
  plt.xticks([])
  #plt.xticks(rotation=90)
  plt.ylabel('|Loss|')
  plt.title('ABS Loss')
  plt.legend()

  plt.tight_layout()
  plt.show()