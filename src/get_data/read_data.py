from binance.client import Client as bnb_client
from datetime import datetime
import pandas as pd
from get_data.read_write_json import read_json, write_json
from tqdm import tqdm
import os
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

client = bnb_client()


def get_binance_px(symbol, freq, start_ts, end_ts):
    data = client.get_historical_klines(symbol, freq, start_ts, end_ts)

    if not data:
        raise ValueError(f"No data returned for {symbol}")

    columns = [
        'open_time','open','high','low','close','volume',
        'close_time','quote_volume','num_trades',
        'taker_base_volume','taker_quote_volume','ignore'
    ]

    df = pd.DataFrame(data, columns=columns)

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

    df['close'] = df['close'].astype(float)

    return df[['open_time', 'close']]


def get_rets(freq='4h', start_ts='2020-01-01', end_ts='2022-12-31'):

    file_path = f"src/misc/valid_tickers_{start_ts[:4]}_to_{end_ts[:4]}.json"

    if os.path.exists(file_path):
        tickers = read_json(file_path)
        logger.info(f"Loaded {len(tickers)} tickers from cached file.")
    else:
        tickers = read_json("src/misc/tickers.json")
        logger.info(f"Loaded {len(tickers)} tickers from base file.")

    tickers = list(dict.fromkeys(["BTCUSDT"] + tickers))

    px_dict = {}
    valid_tickers = []

    for symbol in tqdm(tickers):
        try:
            df = get_binance_px(symbol, freq, start_ts, end_ts)
            df = df.set_index('open_time')
            px_dict[symbol] = df['close']
            valid_tickers.append(symbol)

            # Optional: small sleep to reduce rate-limit risk
            time.sleep(0.1)

        except Exception as e:
            logger.warning(f"Failed for {symbol}: {e}")

    if not px_dict:
        raise ValueError("No valid price data retrieved.")

    # Combine prices
    px = pd.DataFrame(px_dict)

    # Keep only timestamps common across all assets
    px = px.dropna(thresh=int(0.8 * len(px.columns)))
    px = px.dropna(axis=1, thresh=int(0.8 * len(px)))
    valid_tickers = px.columns.tolist()

    logger.info(f"Final universe size: {len(valid_tickers)}")
    logger.info(f"Number of timestamps: {len(px)}")

    # Compute returns safely
    px_filled = px.ffill(limit=1)
    ret = px_filled.pct_change()
    ret[px.isna()] = float('nan')  # re-mask where original had no data
    ret = ret.dropna(how='all')

    # Save stable universe if not cached
    if not os.path.exists(file_path):
        write_json(file_path, valid_tickers)
        logger.info("Saved validated ticker universe.")

    return ret, px