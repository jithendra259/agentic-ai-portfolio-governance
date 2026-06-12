# Integration Patches for Technical Analysis System

## 1️⃣ Update `src/orchestrator/intent_router.py`

### Add these imports at the top:
```python
from src.agents.technical_analysis_a5 import (
    calculate_rsi_signals,
    calculate_macd_crossovers,
    calculate_bollinger_bands,
    detect_support_resistance,
    analyze_trends,
    generate_technical_report,
)
```

### Add these handlers to `_default_handlers()` method:
Find the line `return {` in `_default_handlers()` and add these entries:
```python
            # Technical Analysis Tools
            "calculate_rsi_signals": calculate_rsi_signals,
            "calculate_macd_crossovers": calculate_macd_crossovers,
            "calculate_bollinger_bands": calculate_bollinger_bands,
            "detect_support_resistance": detect_support_resistance,
            "analyze_trends": analyze_trends,
            "generate_technical_report": generate_technical_report,
```

### Add these handlers to the `handle()` method:
Find `if intent_match.intent == IntentType.GREETING:` and add these before/after similar patterns:

```python
        if intent_match.intent == IntentType.TA_PLOT_RSI:
            return self._success(
                intent_match,
                self._invoke("calculate_rsi_signals", {"ticker": intent_match.parameters.get("ticker", "")}),
            )

        if intent_match.intent == IntentType.TA_PLOT_MACD:
            return self._success(
                intent_match,
                self._invoke("calculate_macd_crossovers", {"ticker": intent_match.parameters.get("ticker", "")}),
            )

        if intent_match.intent == IntentType.TA_PLOT_BOLLINGER_BANDS:
            return self._success(
                intent_match,
                self._invoke("calculate_bollinger_bands", {"ticker": intent_match.parameters.get("ticker", "")}),
            )

        if intent_match.intent == IntentType.TA_SUPPORT_RESISTANCE:
            return self._success(
                intent_match,
                self._invoke("detect_support_resistance", {"ticker": intent_match.parameters.get("ticker", "")}),
            )

        if intent_match.intent == IntentType.TA_TREND_ANALYSIS:
            return self._success(
                intent_match,
                self._invoke("analyze_trends", {"ticker": intent_match.parameters.get("ticker", "")}),
            )

        if intent_match.intent == IntentType.TA_FULL_DASHBOARD:
            # Return all 5 indicators in parallel
            ticker = intent_match.parameters.get("ticker", "")
            return self._success(
                intent_match,
                {
                    "rsi": self._invoke("calculate_rsi_signals", {"ticker": ticker}),
                    "macd": self._invoke("calculate_macd_crossovers", {"ticker": ticker}),
                    "bollinger_bands": self._invoke("calculate_bollinger_bands", {"ticker": ticker}),
                    "support_resistance": self._invoke("detect_support_resistance", {"ticker": ticker}),
                    "trend": self._invoke("analyze_trends", {"ticker": ticker}),
                }
            )

        if intent_match.intent == IntentType.TA_TECHNICAL_REPORT:
            return self._success(
                intent_match,
                self._invoke("generate_technical_report", {"ticker": intent_match.parameters.get("ticker", "")}),
            )
```

---

## 2️⃣ Update `src/orchestrator/chatbot_orchestrator.py`

### Add import:
```python
from src.agents.technical_analysis_a5 import TECHNICAL_ANALYSIS_TOOLS
```

### Find the section where agent tools are defined:
Look for something like:
```python
tools = [
    # existing tools...
]
```

### Add the technical analysis tools:
```python
tools = [
    # ... existing tools ...
] + TECHNICAL_ANALYSIS_TOOLS
```

---

## 3️⃣ Create API Endpoints (Optional, for direct tool access)

Add this to `backend/api/main.py`:

```python
from src.agents.technical_analysis_a5 import (
    calculate_rsi_signals,
    calculate_macd_crossovers,
    calculate_bollinger_bands,
    detect_support_resistance,
    analyze_trends,
    generate_technical_report,
)

# Add these endpoints to FastAPI
@app.post("/api/technical-analysis/{ticker}/rsi")
async def api_calculate_rsi(ticker: str, days: int = 252):
    """Calculate RSI for a given ticker."""
    return await calculate_rsi_signals(ticker=ticker.upper(), data_days=days)

@app.post("/api/technical-analysis/{ticker}/macd")
async def api_calculate_macd(ticker: str, days: int = 252):
    """Calculate MACD for a given ticker."""
    return await calculate_macd_crossovers(ticker=ticker.upper(), data_days=days)

@app.post("/api/technical-analysis/{ticker}/bollinger-bands")
async def api_calculate_bb(ticker: str, days: int = 252):
    """Calculate Bollinger Bands for a given ticker."""
    return await calculate_bollinger_bands(ticker=ticker.upper(), data_days=days)

@app.post("/api/technical-analysis/{ticker}/support-resistance")
async def api_detect_sr(ticker: str, days: int = 252):
    """Detect support and resistance levels."""
    return await detect_support_resistance(ticker=ticker.upper(), data_days=days)

@app.post("/api/technical-analysis/{ticker}/trend")
async def api_analyze_trend(ticker: str, days: int = 252):
    """Analyze trend for a given ticker."""
    return await analyze_trends(ticker=ticker.upper(), data_days=days)

@app.post("/api/technical-analysis/{ticker}/full-report")
async def api_full_report(ticker: str, days: int = 252):
    """Generate full technical analysis report."""
    return await generate_technical_report(ticker=ticker.upper(), data_days=days)

@app.get("/api/technical-analysis/{ticker}/dashboard")
async def api_dashboard(ticker: str, days: int = 252):
    """Get data for full technical dashboard (all indicators)."""
    return {
        "ticker": ticker.upper(),
        "rsi": await calculate_rsi_signals(ticker=ticker.upper(), data_days=days),
        "macd": await calculate_macd_crossovers(ticker=ticker.upper(), data_days=days),
        "bollinger_bands": await calculate_bollinger_bands(ticker=ticker.upper(), data_days=days),
        "support_resistance": await detect_support_resistance(ticker=ticker.upper(), data_days=days),
        "trend": await analyze_trends(ticker=ticker.upper(), data_days=days),
    }
```

---

## 4️⃣ Frontend Integration Skeleton

### Create `frontend/src/components/TechnicalDashboard.jsx`:

```jsx
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import Plotly from 'react-plotly.js';

export default function TechnicalDashboard({ ticker, daysBack = 252 }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(
          `/api/technical-analysis/${ticker}/dashboard?days=${daysBack}`
        );
        if (!response.ok) throw new Error(`Failed to fetch data: ${response.status}`);
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [ticker, daysBack]);

  if (loading) return <div className="p-4">Loading technical analysis...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!data) return <div className="p-4">No data available</div>;

  return (
    <div className="w-full space-y-4">
      <div className="text-2xl font-bold">{ticker} - Technical Analysis</div>
      
      <Tabs defaultValue="price" className="w-full">
        <TabsList className="grid w-full grid-cols-8">
          <TabsTrigger value="price">Price</TabsTrigger>
          <TabsTrigger value="volume">Volume</TabsTrigger>
          <TabsTrigger value="ma">MAs</TabsTrigger>
          <TabsTrigger value="rsi">RSI</TabsTrigger>
          <TabsTrigger value="macd">MACD</TabsTrigger>
          <TabsTrigger value="bb">BB</TabsTrigger>
          <TabsTrigger value="sr">S&R</TabsTrigger>
          <TabsTrigger value="trend">Trend</TabsTrigger>
        </TabsList>

        {/* Tab 1: Price + Candlestick */}
        <TabsContent value="price">
          <Card>
            <CardHeader>
              <CardTitle>Price Movement</CardTitle>
            </CardHeader>
            <CardContent>
              {data.trend && (
                <Plotly
                  data={[
                    {
                      x: data.trend.data.map(d => d.date),
                      y: data.trend.data.map(d => d.close),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Close Price',
                    },
                  ]}
                  layout={{
                    title: `${ticker} Price Chart`,
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Price ($)' },
                  }}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 4: RSI */}
        <TabsContent value="rsi">
          <Card>
            <CardHeader>
              <CardTitle>Relative Strength Index (RSI)</CardTitle>
              <div className="text-sm text-gray-500">
                Current: {data.rsi.current_value.toFixed(2)} ({data.rsi.zone})
              </div>
            </CardHeader>
            <CardContent>
              {data.rsi.data && (
                <Plotly
                  data={[
                    {
                      x: data.rsi.data.map(d => d.date),
                      y: data.rsi.data.map(d => d.rsi),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'RSI',
                    },
                    {
                      x: [data.rsi.data[0].date, data.rsi.data[data.rsi.data.length - 1].date],
                      y: [70, 70],
                      mode: 'lines',
                      name: 'Overbought',
                      line: { dash: 'dash', color: 'red' },
                    },
                    {
                      x: [data.rsi.data[0].date, data.rsi.data[data.rsi.data.length - 1].date],
                      y: [30, 30],
                      mode: 'lines',
                      name: 'Oversold',
                      line: { dash: 'dash', color: 'green' },
                    },
                  ]}
                  layout={{
                    title: 'RSI (14)',
                    yaxis: { range: [0, 100] },
                  }}
                />
              )}
              
              {/* Signals List */}
              {data.rsi.signals && data.rsi.signals.length > 0 && (
                <div className="mt-4 space-y-2">
                  <h4 className="font-semibold">Recent Signals:</h4>
                  {data.rsi.signals.slice(-5).map((signal, i) => (
                    <div key={i} className="text-sm border-l-2 border-blue-500 pl-2">
                      <div className="font-medium">{signal.type}</div>
                      <div className="text-gray-600">{signal.evidence}</div>
                      <div className="text-gray-500 text-xs">{signal.governance_narrative}</div>
                      <div className="text-gray-500 text-xs">Confidence: {(signal.confidence * 100).toFixed(0)}%</div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 5: MACD */}
        <TabsContent value="macd">
          <Card>
            <CardHeader>
              <CardTitle>MACD</CardTitle>
              <div className="text-sm text-gray-500">
                Momentum: {data.macd.momentum_direction}
              </div>
            </CardHeader>
            <CardContent>
              {data.macd.data && (
                <Plotly
                  data={[
                    {
                      x: data.macd.data.map(d => d.date),
                      y: data.macd.data.map(d => d.macd),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'MACD',
                    },
                    {
                      x: data.macd.data.map(d => d.date),
                      y: data.macd.data.map(d => d.signal),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Signal',
                    },
                    {
                      x: data.macd.data.map(d => d.date),
                      y: data.macd.data.map(d => d.histogram),
                      type: 'bar',
                      name: 'Histogram',
                    },
                  ]}
                  layout={{
                    title: 'MACD (12, 26, 9)',
                  }}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 6: Bollinger Bands */}
        <TabsContent value="bb">
          <Card>
            <CardHeader>
              <CardTitle>Bollinger Bands</CardTitle>
              <div className="text-sm text-gray-500">
                Squeeze: {data.bollinger_bands.squeeze_detected ? 'Yes' : 'No'}
              </div>
            </CardHeader>
            <CardContent>
              {data.bollinger_bands.data && (
                <Plotly
                  data={[
                    {
                      x: data.bollinger_bands.data.map(d => d.date),
                      y: data.bollinger_bands.data.map(d => d.close),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Price',
                    },
                    {
                      x: data.bollinger_bands.data.map(d => d.date),
                      y: data.bollinger_bands.data.map(d => d.upper_band),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Upper Band',
                      line: { color: 'rgba(255, 0, 0, 0.3)' },
                    },
                    {
                      x: data.bollinger_bands.data.map(d => d.date),
                      y: data.bollinger_bands.data.map(d => d.lower_band),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Lower Band',
                      line: { color: 'rgba(0, 255, 0, 0.3)' },
                    },
                  ]}
                  layout={{
                    title: 'Bollinger Bands (20, 2)',
                  }}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 7: Support & Resistance */}
        <TabsContent value="sr">
          <Card>
            <CardHeader>
              <CardTitle>Support & Resistance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">Resistance</h4>
                  {data.support_resistance.resistance_levels.map((level, i) => (
                    <div key={i} className="text-sm py-1">
                      ${level.toFixed(2)}
                    </div>
                  ))}
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Support</h4>
                  {data.support_resistance.support_levels.map((level, i) => (
                    <div key={i} className="text-sm py-1">
                      ${level.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 8: Trend */}
        <TabsContent value="trend">
          <Card>
            <CardHeader>
              <CardTitle>Trend Analysis</CardTitle>
              <div className="text-sm text-gray-500">
                Trend: {data.trend.summary.trend} (Strength: {(data.trend.summary.alignment_quality * 100).toFixed(0)}%)
              </div>
            </CardHeader>
            <CardContent>
              {data.trend.data && (
                <Plotly
                  data={[
                    {
                      x: data.trend.data.map(d => d.date),
                      y: data.trend.data.map(d => d.close),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Price',
                    },
                    {
                      x: data.trend.data.map(d => d.date),
                      y: data.trend.data.map(d => d.sma_20),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'SMA 20',
                    },
                    {
                      x: data.trend.data.map(d => d.date),
                      y: data.trend.data.map(d => d.sma_50),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'SMA 50',
                    },
                    {
                      x: data.trend.data.map(d => d.date),
                      y: data.trend.data.map(d => d.sma_200),
                      type: 'scatter',
                      mode: 'lines',
                      name: 'SMA 200',
                    },
                  ]}
                  layout={{
                    title: 'Trend Analysis (SMA 20/50/200)',
                  }}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

---

## Testing Integration

### Test 1: Verify Intent Classification
```bash
python3 << 'EOF'
from src.intent.intent_classifier import IntentClassifier

classifier = IntentClassifier()
test_queries = [
    "Plot RSI for AAPL",
    "Show MACD for NVDA",
    "Bollinger Bands on TSLA",
    "Identify support and resistance for SPY",
]

for query in test_queries:
    match = classifier.classify(query)
    print(f"Query: {query}")
    print(f"  Intent: {match.intent.value}")
    print(f"  Confidence: {match.confidence}")
    print(f"  Parameters: {match.parameters}\n")
EOF
```

### Test 2: Verify Tool Execution
```bash
python3 << 'EOF'
from src.agents.technical_analysis_a5 import calculate_rsi_signals
import asyncio

# Test RSI calculation
result = calculate_rsi_signals(ticker="AAPL")
print("RSI Result:")
print(f"  Ticker: {result.get('ticker')}")
print(f"  Current RSI: {result.get('current_value')}")
print(f"  Zone: {result.get('zone')}")
print(f"  Signals: {len(result.get('signals', []))}")
EOF
```

### Test 3: Manual Chatbot Testing
1. Start backend: `python backend/api/main.py`
2. Send message: "Plot RSI for AAPL"
3. Verify intent route to `calculate_rsi_signals()`
4. Check response structure matches tool output

---

## Troubleshooting

### Issue: ModuleNotFoundError for technical_indicators
**Solution**: Ensure `__init__.py` files exist in parent directories

### Issue: Intent not matching
**Solution**: Check `IntentClassifier.TECHNICAL_ANALYSIS_PATTERNS` is in `_compile_patterns()`

### Issue: Tool not executing
**Solution**: Verify import in `intent_router.py` and tool name in handler dictionary

---

**Once these 4 patches are applied, the system will be fully operational!**
