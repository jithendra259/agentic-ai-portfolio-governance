import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate the data structure I just implemented
data = {
    "tickers_included": ["AAPL", "MSFT"],
    "returns": {
        "AAPL": [0.01, -0.02, 0.03, 0.01, -0.01],
        "MSFT": [0.02, -0.01, 0.02, 0.02, -0.02]
    },
    "return_dates": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]
}

# Simulate the AI code that WOULD be generated now with the new rules
code = """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# RULE 7: Use return_dates index
df = pd.DataFrame(data['returns'], index=pd.to_datetime(data['return_dates']))

plt.style.use("dark_background")
plt.rcParams.update({"figure.facecolor": "#0b1020", "axes.facecolor": "#111827",
                    "axes.labelcolor": "#e5e7eb", "xtick.color": "#d1d5db",
                    "ytick.color": "#d1d5db", "text.color": "#f3f4f6"})

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, ax=ax)
ax.set_title("Test Plot (Correct Dates)")

# Check the index to verify it's NOT 1970
print(f"DEBUG INDEX: {df.index[:2]}")

plt.savefig("outputs/verification_test.png")
plt.close()
"""

# Execute locally
exec(code, {"data": data})
print("Verification script finished. check outputs/verification_test.png")
