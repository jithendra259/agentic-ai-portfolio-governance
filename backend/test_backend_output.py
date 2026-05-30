import json
import asyncio
from src.agents.live_data_tools import plot_historical_prices
from src.agents.plot_store import GLOBAL_PLOT_DATA

def main():
    print("Fetching data from backend tool...")
    tool_args = {
        "tickers": ["AAPL", "MSFT"],
        "start_date": "2020-01-01",
        "end_date": "2024-12-31"
    }
    config = {"configurable": {"thread_id": "test_output"}}
    
    # Clear store for testing
    GLOBAL_PLOT_DATA.clear()
    
    # Run the tool directly
    try:
        result = plot_historical_prices.invoke(tool_args, config=config)
        print("\n=== TOOL TEXT RESPONSE ===")
        print(result)
        
        if "test_output" in GLOBAL_PLOT_DATA:
            plot_spec = GLOBAL_PLOT_DATA["test_output"]
            with open("plot_output.json", "w") as f:
                json.dump(plot_spec, f, indent=2, default=str)
            print("\nSUCCESS! The JSON data has been saved to: plot_output.json")
        else:
            print("No data was stored in GLOBAL_PLOT_DATA!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
