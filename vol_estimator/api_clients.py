"""API clients for financial data"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime, timedelta


class AlphaVantageClient:
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.last_call_time = 0
        self.min_call_interval = 12
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_daily_data(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        self._rate_limit()
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "datatype": "csv"
        }
        
        if self.api_key:
            params["apikey"] = self.api_key
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = [col.replace(' ', '_') for col in df.columns]
            
            return df
        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()


class CoinGeckoClient:
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self.last_call_time = 0
        self.min_call_interval = 1.2
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_historical_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        self._rate_limit()
        
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices = data.get('prices', [])
            if not prices:
                return pd.DataFrame()
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.resample('D').last()
            
            return df
        except Exception as e:
            print(f"Error fetching CoinGecko data for {coin_id}: {e}")
            return pd.DataFrame()


class FinnhubClient:
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.last_call_time = 0
        self.min_call_interval = 1.0
        self._sector_cache: Dict[str, str] = {}
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        if not self.api_key:
            return None
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}/stock/profile2"
        params = {
            "symbol": symbol,
            "token": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data or not data:
                return None
            
            return data
        except Exception as e:
            print(f"Error fetching Finnhub profile for {symbol}: {e}")
            return None
    
    def get_sector(self, symbol: str) -> Optional[str]:
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        
        profile = self.get_company_profile(symbol)
        if profile and 'finnhubIndustry' in profile:
            sector = profile['finnhubIndustry']
            self._sector_cache[symbol] = sector
            return sector
        
        return None
    
    def get_sectors_batch(self, symbols: List[str]) -> Dict[str, str]:
        sectors = {}
        
        for symbol in symbols:
            sector = self.get_sector(symbol)
            if sector:
                sectors[symbol] = sector
        
        return sectors


class YahooFinanceClient:
    
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            print(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()


class FinancialDataFetcher:
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, finnhub_key: Optional[str] = None):
        self.av_client = AlphaVantageClient(alpha_vantage_key)
        self.cg_client = CoinGeckoClient()
        self.yf_client = YahooFinanceClient()
        self.finnhub_client = FinnhubClient(finnhub_key)
        
        self.crypto_symbols = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'SOL': 'solana',
            'XRP': 'ripple',
            'DOT': 'polkadot',
            'DOGE': 'dogecoin',
            'MATIC': 'matic-network',
            'LTC': 'litecoin'
        }
    
    def fetch_data(self, symbols: List[str], period: str = "1y", source: str = "auto") -> Dict[str, pd.DataFrame]:
        data = {}
        
        for symbol in symbols:
            if source == "auto":
                if symbol.upper() in self.crypto_symbols:
                    df = self._fetch_crypto(symbol)
                else:
                    df = self.yf_client.get_historical_data(symbol, period=period)
                    if df.empty:
                        df = self.av_client.get_daily_data(symbol)
            elif source == "yahoo":
                df = self.yf_client.get_historical_data(symbol, period=period)
            elif source == "alphavantage":
                df = self.av_client.get_daily_data(symbol)
            elif source == "coingecko":
                df = self._fetch_crypto(symbol)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def _fetch_crypto(self, symbol: str) -> pd.DataFrame:
        symbol_upper = symbol.upper()
        if symbol_upper in self.crypto_symbols:
            coin_id = self.crypto_symbols[symbol_upper]
            df = self.cg_client.get_historical_data(coin_id, days=365)
            if not df.empty:
                if 'price' in df.columns:
                    df.rename(columns={'price': 'Close'}, inplace=True)
        else:
            if '-' not in symbol:
                symbol = f"{symbol}-USD"
            df = self.yf_client.get_historical_data(symbol)
        
        return df
    
    def compute_returns(self, data: Dict[str, pd.DataFrame], method: str = "log") -> Dict[str, np.ndarray]:
        returns = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            if 'Close' in df.columns:
                prices = df['Close'].values
            elif 'close' in df.columns:
                prices = df['close'].values
            else:
                continue
            
            if method == "log":
                ret = np.diff(np.log(prices))
            else:
                ret = np.diff(prices) / prices[:-1]
            
            returns[symbol] = ret
        
        return returns
    
    def align_returns(self, returns: Dict[str, np.ndarray], min_length: int = 100) -> Tuple[np.ndarray, List[str]]:
        valid_returns = {
            sym: ret for sym, ret in returns.items() 
            if len(ret) >= min_length
        }
        
        if not valid_returns:
            return np.array([]), []
        
        min_len = min(len(ret) for ret in valid_returns.values())
        
        aligned = []
        symbols = []
        for symbol, ret in valid_returns.items():
            aligned.append(ret[-min_len:])
            symbols.append(symbol)
        
        aligned_matrix = np.array(aligned)
        
        return aligned_matrix, symbols
    
    def get_sectors(self, symbols: List[str], source: str = "auto") -> Dict[str, str]:
        sectors = {}
        
        for symbol in symbols:
            if self._is_crypto(symbol):
                sectors[symbol] = "Cryptocurrency"
                continue
            
            sector = None
            
            if self.finnhub_client.api_key and source in ("auto", "finnhub"):
                sector = self.finnhub_client.get_sector(symbol)
            
            if sector is None and source in ("auto", "yfinance"):
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    sector = info.get('sector')
                except Exception:
                    pass
            
            sectors[symbol] = sector or "Unknown"
        
        return sectors
    
    def _is_crypto(self, symbol: str) -> bool:
        return symbol.endswith('-USD') or symbol.upper() in self.crypto_symbols

