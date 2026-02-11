from binance.client import Client as bnb_client
from datetime import datetime
import pandas as pd 
from src.misc.read_write_json import read_json, write_json
from tqdm import tqdm

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)


tickers = read_json("src/misc/tickers.json")

#client = bnb_client()
###  if you're in the US, use: 
client = bnb_client(tld='US')#" here instead

def get_binance_px(symbol,freq,start_ts,end_ts):
    data = client.get_historical_klines(symbol,freq,start_ts,end_ts)
    columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
    'num_trades','taker_base_volume','taker_quote_volume','ignore']

    data = pd.DataFrame(data,columns = columns)
    
    # Convert from POSIX timestamp (number of millisecond since jan 1, 1970)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms', utc=True)
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms', utc=True)
    return data 


def get_rets(freq='4h',start_ts = '2020-01-01',end_ts='2022-12-31'):

    px = {}
    for x in tqdm(tickers):
        try:
            data = get_binance_px(x,freq,start_ts,end_ts)
            px[x] = data.set_index('open_time')['close']
        except:
            tickers.remove(x)

    px = pd.DataFrame(px).astype(float)
    px = px.reindex(pd.date_range(px.index[0],px.index[-1],freq=freq))
    ret = px.pct_change().dropna(how="all",axis=0).dropna(how="all",axis=1)

    write_json(f"src/misc/valid_tickers_{start_ts[:4]}_to_{end_ts[:4]}.json", ret.columns.tolist())

    return ret