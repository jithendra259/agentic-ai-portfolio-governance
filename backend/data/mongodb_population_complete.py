#!/usr/bin/env python3
"""
🚀 MongoDB Population Script - COMPLETE GRAPH-RAG + TIME-SERIES VERSION
Fetches yfinance data for 220 tickers and stores in MongoDB Atlas.
Includes 2005-2025 Historical Prices, relationships, financials, stealth delays, and NaT/NaN cleaning.
Includes `certifi` + `ssl.PROTOCOL_TLSv1_2` patch to force secure Atlas connections.
"""

import yfinance as yf
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import logging
import os
import time
import random
import certifi
import ssl  # 🚨 Added to force TLS 1.2
from pathlib import Path
from dotenv import load_dotenv

# =====================================================
# SETUP LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# LOAD ENVIRONMENT & CONFIG
# =====================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

def normalize_mongo_uri(raw_uri):
    if not raw_uri: raise ValueError("MONGO_URI not found in .env file")
    uri = raw_uri.strip().strip("\"'").replace(" ", "")
    if "mongodb+srv://mongodb+srv://" in uri:
        uri = uri.replace("mongodb+srv://mongodb+srv://", "mongodb+srv://", 1)
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        raise ValueError("MONGO_URI must start with 'mongodb://' or 'mongodb+srv://'")
    return uri

MONGO_URI = normalize_mongo_uri(os.getenv("MONGO_URI"))
DB_NAME = "Stock_data"
COLLECTION_NAME = "ticker"
UNIVERSE_COOLDOWN_SECONDS = 8

UNIVERSES = {
    "U1": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "CSCO", "QCOM", "ORCL",
           "INTU", "AMAT", "IBM", "TXN", "NOW", "MU", "PANW", "LRCX", "ADI", "KLAC"],
    "U2": ["JPM", "BAC", "GS", "WFC", "BLK", "SCHW", "FIS", "AXP", "PNC", "TD",
           "COF", "USB", "FITB", "CFG", "RF", "TROW", "JEF", "MU", "EVR", "RJF"],
    "U3": ["JNJ", "PFE", "UNH", "MRK", "ABBV", "AMGN", "LLY", "GILD", "CVLT", "REGN",
           "BIOPSY", "EXAS", "VEEV", "CAR", "DXCM", "VRTX", "ALKS", "CRSP", "NYCB", "ALCO"],
    "U4": ["AMZN", "TSLA", "MCD", "NKE", "SBUX", "ADIDAS", "ASML", "DHI", "ROP", "POOL",
           "LVS", "MGM", "WYNN", "CCL", "RCL", "UAL", "DAL", "AAL", "ZM", "DASH"],
    "U5": ["BA", "CAT", "GE", "HON", "LMT", "RTX", "RAYTHEON", "EMR", "ABB", "ITT",
           "ETN", "IR", "IEX", "JCI", "FLEX", "DOV", "PH", "RSG", "SNA", "SPX"],
    "U6": ["XOM", "CVX", "COP", "MPC", "PSX", "VLO", "EQNR", "HES", "FANG", "EOG",
           "OXY", "SLB", "HAL", "CHD", "ADNCY", "REPYY", "PBA", "BRK", "ENB", "TRP"],
    "U7": ["KO", "PG", "WMT", "COST", "MO", "PM", "UL", "CLX", "SJM", "HSY",
           "MNST", "KDP", "PEP", "CHD", "TSN", "MDLZ", "GIS", "KMB", "TAP", "STZ"],
    "U8": ["NEE", "DUK", "SO", "EXC", "AEP", "EQT", "PEG", "ETR", "WEC", "CMS",
           "XEL", "EVRG", "AES", "AWK", "NRG", "TRGP", "ES", "FLR", "WEE", "LNT"],
    "U9": ["AMT", "PLD", "CCI", "EQIX", "DLR", "ARE", "PSA", "SPG", "IRM", "REG",
           "EGP", "OKE", "LTC", "PK", "STAG", "COLD", "WELL", "EXR", "AKR", "NHI"],
    "U10": ["LIN", "FCX", "NUE", "X", "SCCO", "AA", "ALB", "CDE", "SQM", "CMPR",
            "CF", "APD", "EMN", "LYB", "DOW", "DD", "MOS", "IFF", "WLK", "PPG"],
    "U11": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "CHTR", "T", "VZ", "TMUS", "S",
            "DISH", "MKSI", "QTRX", "IPHI", "ASX", "CPRT", "TTWO", "ATVI", "EA", "RBLX"]
}

# =====================================================
# MONGODB CONNECTION
# =====================================================
def connect_mongodb():
    try:
        logger.info("🔗 Connecting to MongoDB Atlas (Bypassing SSL Interception)...")
        # 🚨 THE MAGIC FIX: tlsAllowInvalidCertificates=True forces Python to ignore Antivirus/ISP blocks
        client = MongoClient(
            MONGO_URI, 
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsAllowInvalidCertificates=True 
        )
        client.admin.command("ping")
        logger.info("✅ MongoDB Connection Successful!")
        return client[DB_NAME][COLLECTION_NAME]
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise

# =====================================================
# DATA EXTRACTION HELPERS
# =====================================================
def fetch_safe(fetcher, default):
    """Safely fetch data and return default if Yahoo fails."""
    try:
        data = fetcher()
        return data if data is not None else default
    except Exception:
        return default

def clean_df_for_mongo(df):
    """Safely converts DataFrame to dict, handling NaT and NaN for MongoDB."""
    if df is None or df.empty:
        return []
    
    # Convert datetime columns to string (Y-m-d) to prevent PyMongo NaT errors
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            
    # Replace all pandas NaT/NaN with Python None (which becomes 'null' in MongoDB)
    df_clean = df.astype(object).where(pd.notnull(df), None)
    return df_clean.to_dict('records')

def extract_financials(ticker):
    """Extracts full financial DataFrames into JSON dictionaries."""
    def df_to_clean_dict(df):
        if df is None or df.empty: return []
        data = []
        for col in df.columns[:5]: # Last 5 periods
            try:
                metrics = df[col].dropna().to_dict()
                data.append({
                    "date": col.strftime("%Y-%m-%d") if pd.notnull(col) else None,
                    "metrics": metrics
                })
            except: pass
        return data

    return {
        "income_statement": {
            "annual": df_to_clean_dict(fetch_safe(lambda: ticker.income_stmt, pd.DataFrame())),
            "quarterly": df_to_clean_dict(fetch_safe(lambda: ticker.quarterly_income_stmt, pd.DataFrame()))
        },
        "balance_sheet": {
            "annual": df_to_clean_dict(fetch_safe(lambda: ticker.balance_sheet, pd.DataFrame())),
            "quarterly": df_to_clean_dict(fetch_safe(lambda: ticker.quarterly_balance_sheet, pd.DataFrame()))
        },
        "cashflow": {
            "annual": df_to_clean_dict(fetch_safe(lambda: ticker.cashflow, pd.DataFrame())),
            "quarterly": df_to_clean_dict(fetch_safe(lambda: ticker.quarterly_cashflow, pd.DataFrame()))
        }
    }

def extract_relationships_and_corp(ticker):
    """Extracts nodes and edges for Graph RAG mapping."""
    data = {
        "dividends": [], "splits": [], "major_holders": [], 
        "institutional_holders": [], "insider_roster": [], "insider_transactions": []
    }
    
    # Corp Actions
    divs = fetch_safe(lambda: ticker.dividends, pd.Series(dtype=float))
    if not divs.empty:
        data["dividends"] = [{"date": k.strftime("%Y-%m-%d"), "amount": float(v)} for k, v in divs.tail(5).items() if pd.notnull(k)]
        
    splits = fetch_safe(lambda: ticker.splits, pd.Series(dtype=float))
    if not splits.empty:
        data["splits"] = [{"date": k.strftime("%Y-%m-%d"), "ratio": f"{int(v)}:1"} for k, v in splits.tail(5).items() if pd.notnull(k)]

    # Holders & Insiders
    majors = fetch_safe(lambda: ticker.major_holders, pd.DataFrame())
    data["major_holders"] = clean_df_for_mongo(majors)

    inst = fetch_safe(lambda: ticker.institutional_holders, pd.DataFrame())
    data["institutional_holders"] = clean_df_for_mongo(inst.head(10) if not inst.empty else inst)

    roster = fetch_safe(lambda: ticker.insider_roster_holders, pd.DataFrame())
    data["insider_roster"] = clean_df_for_mongo(roster.head(10) if not roster.empty else roster)

    trades = fetch_safe(lambda: ticker.insider_transactions, pd.DataFrame())
    data["insider_transactions"] = clean_df_for_mongo(trades.head(10) if not trades.empty else trades)

    return data

def extract_analysis(ticker):
    """Extracts analyst recommendations."""
    analysis = {"recommendations": [], "earnings_estimate": []}
    
    rec_df = fetch_safe(lambda: ticker.recommendations, pd.DataFrame())
    analysis["recommendations"] = clean_df_for_mongo(rec_df.tail(5) if not rec_df.empty else rec_df)
    
    est_df = fetch_safe(lambda: ticker.earnings_estimate, pd.DataFrame())
    analysis["earnings_estimate"] = clean_df_for_mongo(est_df)
    
    return analysis

# =====================================================
# BUILD COMPLETE DOCUMENT
# =====================================================
def build_document(symbol, universe):
    try:
        logger.debug(f"  Fetching {symbol} from yfinance...")
        ticker = yf.Ticker(symbol) 
        info = fetch_safe(lambda: ticker.info, {})
        
        # Fetch 2005-2025 Historical Prices for Agent 1 (Time-Series)
        hist_df = fetch_safe(lambda: ticker.history(start="2005-01-01", end="2025-12-31"), pd.DataFrame())
        historical_prices = []
        if not hist_df.empty:
            hist_df = hist_df.reset_index() # Move 'Date' from index to column
            # Drop columns we don't need for the optimizer to save DB space
            hist_df = hist_df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
            # Use our robust cleaner to handle NaT/NaN and convert to dict
            historical_prices = clean_df_for_mongo(hist_df)
        
        doc = {
            "ticker": symbol,
            "universes": [universe],
            
            "historical_prices": historical_prices, # 📈 Time-Series data injected here
            
            "info": {
                "company_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "city": info.get("city"),
                "summary": info.get("longBusinessSummary"),
                "website": info.get("website")
            },
            
            "key_stats": {
                "market_cap": info.get("marketCap"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "profit_margin": info.get("profitMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta")
            },

            "financials": extract_financials(ticker),
            "graph_relationships": extract_relationships_and_corp(ticker),
            "analysis_and_estimates": extract_analysis(ticker),
            
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        return doc
    except Exception as e:
        logger.error(f"Error building document for {symbol}: {e}")
        return None

# =====================================================
# MAIN POPULATION LOOP
# =====================================================
def populate_mongodb(collection):
    total_tickers = sum(len(tickers) for tickers in UNIVERSES.values())
    processed, failed = 0, []
    
    for universe, tickers in UNIVERSES.items():
        logger.info(f"\n📊 Universe {universe} ({len(tickers)} tickers)")
        
        for index, symbol in enumerate(tickers):
            try:
                doc = build_document(symbol, universe)
                if doc:
                    collection.update_one({"ticker": symbol}, {"$set": doc}, upsert=True)
                    processed += 1
                    logger.info(f"  ✅ {symbol}")
                else:
                    failed.append(symbol)
                    logger.warning(f"  ⚠️ {symbol} (skipped - no data)")
            except Exception as e:
                failed.append(symbol)
                logger.error(f"  ❌ {symbol}: {e}")

            # Stealth delay to avoid 429s (between 3 to 6 seconds)
            if index < len(tickers) - 1:
                sleep_time = random.uniform(3.0, 6.0)
                time.sleep(sleep_time)

        if universe != list(UNIVERSES.keys())[-1]:
            logger.info(f"  Cooling down for {UNIVERSE_COOLDOWN_SECONDS}s...")
            time.sleep(UNIVERSE_COOLDOWN_SECONDS)
            
    logger.info(f"\n✅ Inserted/Updated: {processed}/{total_tickers} | ⚠️ Failed: {len(failed)}")
    return processed, failed

if __name__ == "__main__":
    try:
        collection = connect_mongodb()
        populate_mongodb(collection)
        logger.info("📊 Database is fully populated and ready for Agent 1 and Agent 2.\n")
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}")