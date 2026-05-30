import asyncio
from src.agents.live_data_tools import plot_historical_prices
from src.agents.plot_store import GLOBAL_PLOT_DATA
import json

def test():
    res = plot_historical_prices.invoke(
        {
            'tickers': ['AAPL', 'MSFT'],
            'start_date': '2020-01-01',
            'end_date': '2024-12-31'
        },
        config={"configurable": {"thread_id": "test_id"}}
    )
    if 'test_id' in GLOBAL_PLOT_DATA:
        print(json.dumps(GLOBAL_PLOT_DATA['test_id'][0], default=str, indent=2))

if __name__ == '__main__':
    test()
