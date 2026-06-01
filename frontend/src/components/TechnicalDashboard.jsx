/**
 * TechnicalDashboard.jsx — Bloomberg-Style Technical Analysis Dashboard
 * 
 * 8 interactive tabs for comprehensive technical analysis:
 * 1. Price + Volume (candlestick + volume bars)
 * 2. Volume Analysis (volume with MA)
 * 3. Moving Averages (SMA 20/50/200 alignment)
 * 4. RSI (with overbought/oversold zones)
 * 5. MACD (line + signal + histogram)
 * 6. Bollinger Bands (bands + squeeze detection)
 * 7. Support & Resistance (levels + zones)
 * 8. Trend Analysis (MA lines + alignment)
 * 
 * Includes signal visualization, governance narratives, and recommendations.
 */

import React, { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

export default function TechnicalDashboard({ ticker = 'AAPL', daysBack = 252 }) {
  const [dashboardData, setDashboardData] = useState(null);
  const [fullReport, setFullReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all indicator data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch dashboard data (all 5 indicators in parallel)
        const dashResponse = await fetch(
          `/api/technical-analysis/${ticker.toUpperCase()}/dashboard?days=${daysBack}`
        );
        if (!dashResponse.ok) {
          throw new Error(`Dashboard fetch failed: ${dashResponse.statusText}`);
        }
        const dashData = await dashResponse.json();
        setDashboardData(dashData);

        // Optionally fetch full technical report
        try {
          const reportResponse = await fetch(
            `/api/technical-analysis/${ticker.toUpperCase()}/full-report?days=${daysBack}`
          );
          if (reportResponse.ok) {
            const report = await reportResponse.json();
            setFullReport(report);
          }
        } catch (err) {
          console.warn('Full report fetch failed (non-critical):', err);
        }
      } catch (err) {
        setError(err.message);
        console.error('Dashboard fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    if (ticker) {
      fetchData();
    }
  }, [ticker, daysBack]);

  if (loading) {
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold mb-4">Loading Technical Analysis</div>
          <div className="text-gray-500">Calculating indicators for {ticker}...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Failed to load technical analysis: {error}
        </AlertDescription>
      </Alert>
    );
  }

  if (!dashboardData) {
    return (
      <Alert>
        <AlertDescription>No data available for {ticker}</AlertDescription>
      </Alert>
    );
  }

  const recommendation = fullReport?.recommendations?.primary_recommendation || 'NEUTRAL';
  const confidence = fullReport?.recommendations?.confidence || 0;
  const recommendationColor = {
    BUY: 'bg-green-100 text-green-800',
    SELL: 'bg-red-100 text-red-800',
    NEUTRAL: 'bg-yellow-100 text-yellow-800',
  }[recommendation] || 'bg-gray-100 text-gray-800';

  return (
    <div className="w-full space-y-4 p-4 bg-white">
      {/* Header with Recommendation */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">{ticker} - Technical Analysis</h1>
            <p className="text-sm text-gray-500">
              {dashboardData.trend?.data?.[0]?.date} to{' '}
              {dashboardData.trend?.data?.[dashboardData.trend?.data?.length - 1]?.date}
            </p>
          </div>
          <div className="text-right">
            <div className={`inline-block px-4 py-2 rounded-lg font-bold ${recommendationColor}`}>
              {recommendation}
            </div>
            <div className="text-sm text-gray-600 mt-1">
              Confidence: {(confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Key Metrics Row */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Trend</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold capitalize">
                {dashboardData.trend?.summary?.trend}
              </div>
              <p className="text-xs text-gray-500">
                Strength: {(dashboardData.trend?.summary?.alignment_quality * 100).toFixed(0)}%
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">RSI</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {dashboardData.rsi?.current_value?.toFixed(1)}
              </div>
              <p className="text-xs text-gray-500 capitalize">
                {dashboardData.rsi?.zone}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">MACD</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${
                dashboardData.macd?.momentum_direction === 'bullish' 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`}>
                {dashboardData.macd?.momentum_direction === 'bullish' ? '↑' : '↓'}
              </div>
              <p className="text-xs text-gray-500 capitalize">
                {dashboardData.macd?.momentum_direction}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">BB Squeeze</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {dashboardData.bollinger_bands?.squeeze_detected ? '🔴' : '🟢'}
              </div>
              <p className="text-xs text-gray-500">
                {dashboardData.bollinger_bands?.squeeze_detected ? 'Active' : 'Normal'}
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 8-Tab Dashboard */}
      <Tabs defaultValue="price" className="w-full">
        <TabsList className="grid w-full grid-cols-8 mb-4">
          <TabsTrigger value="price" className="text-xs sm:text-sm">Price</TabsTrigger>
          <TabsTrigger value="volume" className="text-xs sm:text-sm">Volume</TabsTrigger>
          <TabsTrigger value="ma" className="text-xs sm:text-sm">MAs</TabsTrigger>
          <TabsTrigger value="rsi" className="text-xs sm:text-sm">RSI</TabsTrigger>
          <TabsTrigger value="macd" className="text-xs sm:text-sm">MACD</TabsTrigger>
          <TabsTrigger value="bb" className="text-xs sm:text-sm">BB</TabsTrigger>
          <TabsTrigger value="sr" className="text-xs sm:text-sm">S&R</TabsTrigger>
          <TabsTrigger value="trend" className="text-xs sm:text-sm">Trend</TabsTrigger>
        </TabsList>

        {/* TAB 1: PRICE & CANDLESTICK */}
        <TabsContent value="price">
          <Card>
            <CardHeader>
              <CardTitle>Price Movement</CardTitle>
              <CardDescription>Daily candlestick chart</CardDescription>
            </CardHeader>
            <CardContent>
              {dashboardData.trend?.data && (
                <PriceChart data={dashboardData.trend.data} ticker={ticker} />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 2: VOLUME ANALYSIS */}
        <TabsContent value="volume">
          <Card>
            <CardHeader>
              <CardTitle>Volume Analysis</CardTitle>
              <CardDescription>Trading volume with moving average</CardDescription>
            </CardHeader>
            <CardContent>
              {dashboardData.trend?.data && (
                <VolumeChart data={dashboardData.trend.data} ticker={ticker} />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 3: MOVING AVERAGES */}
        <TabsContent value="ma">
          <Card>
            <CardHeader>
              <CardTitle>Moving Averages</CardTitle>
              <CardDescription>SMA 20 / 50 / 200 Alignment</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.trend?.data && (
                <MovingAveragesChart data={dashboardData.trend.data} ticker={ticker} />
              )}
              <MovingAveragesSummary data={dashboardData.trend?.summary} />
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 4: RSI */}
        <TabsContent value="rsi">
          <Card>
            <CardHeader>
              <CardTitle>Relative Strength Index (RSI)</CardTitle>
              <CardDescription>14-period momentum indicator</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.rsi?.data && (
                <RSIChart data={dashboardData.rsi.data} ticker={ticker} />
              )}
              <SignalsList signals={dashboardData.rsi?.signals} title="RSI Signals" />
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 5: MACD */}
        <TabsContent value="macd">
          <Card>
            <CardHeader>
              <CardTitle>MACD (12, 26, 9)</CardTitle>
              <CardDescription>Moving Average Convergence Divergence</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.macd?.data && (
                <MACDChart data={dashboardData.macd.data} ticker={ticker} />
              )}
              <SignalsList signals={dashboardData.macd?.signals} title="MACD Signals" />
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 6: BOLLINGER BANDS */}
        <TabsContent value="bb">
          <Card>
            <CardHeader>
              <CardTitle>Bollinger Bands (20, 2σ)</CardTitle>
              <CardDescription>Volatility and squeeze detection</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.bollinger_bands?.data && (
                <BollingerBandsChart
                  data={dashboardData.bollinger_bands.data}
                  ticker={ticker}
                />
              )}
              <BBSummary summary={dashboardData.bollinger_bands?.summary} />
              <SignalsList signals={dashboardData.bollinger_bands?.signals} title="BB Signals" />
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 7: SUPPORT & RESISTANCE */}
        <TabsContent value="sr">
          <Card>
            <CardHeader>
              <CardTitle>Support & Resistance</CardTitle>
              <CardDescription>Key price levels and zones</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <SRLevels summary={dashboardData.support_resistance?.summary} />
              <SignalsList
                signals={dashboardData.support_resistance?.signals}
                title="Proximity Signals"
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 8: TREND ANALYSIS */}
        <TabsContent value="trend">
          <Card>
            <CardHeader>
              <CardTitle>Trend Analysis</CardTitle>
              <CardDescription>Trend direction and MA crossovers</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.trend?.data && (
                <TrendChart data={dashboardData.trend.data} ticker={ticker} />
              )}
              <SignalsList signals={dashboardData.trend?.signals} title="Trend Signals" />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Full Report Section (if available) */}
      {fullReport && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Technical Analysis Report</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ReportSummary report={fullReport} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// CHART COMPONENTS
// ============================================================================

function PriceChart({ data, ticker }) {
  const chartData = [
    {
      x: data.map(d => d.date),
      y: data.map(d => d.close),
      type: 'scatter',
      mode: 'lines',
      name: 'Close Price',
      line: { color: '#3b82f6', width: 2 },
    },
  ];

  return (
    <Plot
      data={chartData}
      layout={{
        title: `${ticker} Price Chart`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        hovermode: 'x unified',
        height: 400,
      }}
      config={{ responsive: true }}
    />
  );
}

function VolumeChart({ data, ticker }) {
  const volumes = data.map(d => d.volume || 0);
  const avgVolume =
    volumes.reduce((a, b) => a + b, 0) / volumes.length;

  return (
    <Plot
      data={[
        {
          x: data.map(d => d.date),
          y: volumes,
          type: 'bar',
          name: 'Volume',
          marker: { color: '#8b5cf6' },
        },
        {
          x: [data[0].date, data[data.length - 1].date],
          y: [avgVolume, avgVolume],
          mode: 'lines',
          name: 'Avg Volume',
          line: { dash: 'dash', color: 'orange' },
        },
      ]}
      layout={{
        title: `${ticker} Volume Analysis`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Volume' },
        height: 400,
      }}
      config={{ responsive: true }}
    />
  );
}

function MovingAveragesChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: data.map(d => d.date),
          y: data.map(d => d.close),
          type: 'scatter',
          mode: 'lines',
          name: 'Price',
          line: { color: '#1f2937', width: 2 },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma_20),
          type: 'scatter',
          mode: 'lines',
          name: 'SMA 20',
          line: { color: '#f59e0b' },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma_50),
          type: 'scatter',
          mode: 'lines',
          name: 'SMA 50',
          line: { color: '#ef4444' },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma_200),
          type: 'scatter',
          mode: 'lines',
          name: 'SMA 200',
          line: { color: '#6366f1' },
        },
      ]}
      layout={{
        title: `${ticker} Moving Averages (20/50/200)`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        hovermode: 'x unified',
        height: 400,
      }}
      config={{ responsive: true }}
    />
  );
}

function RSIChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: data.map(d => d.date),
          y: data.map(d => d.rsi),
          type: 'scatter',
          mode: 'lines',
          name: 'RSI',
          line: { color: '#06b6d4', width: 2 },
        },
        {
          x: [data[0].date, data[data.length - 1].date],
          y: [70, 70],
          mode: 'lines',
          name: 'Overbought',
          line: { dash: 'dash', color: 'red' },
        },
        {
          x: [data[0].date, data[data.length - 1].date],
          y: [30, 30],
          mode: 'lines',
          name: 'Oversold',
          line: { dash: 'dash', color: 'green' },
        },
      ]}
      layout={{
        title: `${ticker} RSI (14)`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'RSI', range: [0, 100] },
        hovermode: 'x unified',
        height: 400,
        shapes: [
          {
            type: 'rect',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 70,
            y1: 100,
            fillcolor: 'rgba(255, 0, 0, 0.1)',
            layer: 'below',
          },
          {
            type: 'rect',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 0,
            y1: 30,
            fillcolor: 'rgba(0, 255, 0, 0.1)',
            layer: 'below',
          },
        ],
      }}
      config={{ responsive: true }}
    />
  );
}

function MACDChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: data.map(d => d.date),
          y: data.map(d => d.macd),
          type: 'scatter',
          mode: 'lines',
          name: 'MACD',
          line: { color: '#3b82f6' },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.signal),
          type: 'scatter',
          mode: 'lines',
          name: 'Signal',
          line: { color: '#ef4444' },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.histogram),
          type: 'bar',
          name: 'Histogram',
          marker: {
            color: data.map(d =>
              d.histogram >= 0 ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'
            ),
          },
        },
      ]}
      layout={{
        title: `${ticker} MACD (12, 26, 9)`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'MACD' },
        hovermode: 'x unified',
        height: 400,
      }}
      config={{ responsive: true }}
    />
  );
}

function BollingerBandsChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: data.map(d => d.date),
          y: data.map(d => d.close),
          type: 'scatter',
          mode: 'lines',
          name: 'Price',
          line: { color: '#1f2937', width: 2 },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.upper_band),
          type: 'scatter',
          mode: 'lines',
          name: 'Upper Band',
          line: { color: 'rgba(239, 68, 68, 0.5)', dash: 'dash' },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma),
          type: 'scatter',
          mode: 'lines',
          name: 'Middle Band',
          line: { color: '#f59e0b' },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.lower_band),
          type: 'scatter',
          mode: 'lines',
          name: 'Lower Band',
          line: { color: 'rgba(34, 197, 94, 0.5)', dash: 'dash' },
        },
      ]}
      layout={{
        title: `${ticker} Bollinger Bands (20, 2σ)`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        hovermode: 'x unified',
        height: 400,
      }}
      config={{ responsive: true }}
    />
  );
}

function TrendChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: data.map(d => d.date),
          y: data.map(d => d.close),
          type: 'scatter',
          mode: 'lines',
          name: 'Price',
          line: { color: '#1f2937', width: 2 },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma_20),
          type: 'scatter',
          mode: 'lines',
          name: 'SMA 20',
          line: { color: '#f59e0b', width: 2 },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma_50),
          type: 'scatter',
          mode: 'lines',
          name: 'SMA 50',
          line: { color: '#ef4444', width: 2 },
        },
        {
          x: data.map(d => d.date),
          y: data.map(d => d.sma_200),
          type: 'scatter',
          mode: 'lines',
          name: 'SMA 200',
          line: { color: '#6366f1', width: 2 },
        },
      ]}
      layout={{
        title: `${ticker} Trend Analysis`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        hovermode: 'x unified',
        height: 400,
      }}
      config={{ responsive: true }}
    />
  );
}

// ============================================================================
// INFO COMPONENTS
// ============================================================================

function MovingAveragesSummary({ data }) {
  if (!data) return null;

  const trend = data.trend;
  const sma20 = data.sma_20;
  const sma50 = data.sma_50;
  const sma200 = data.sma_200;

  return (
    <div className="grid grid-cols-4 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Current Price</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">${data.current_price?.toFixed(2)}</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">SMA 20</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">${sma20?.toFixed(2)}</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">SMA 50</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">${sma50?.toFixed(2)}</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">SMA 200</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">${sma200?.toFixed(2)}</div>
        </CardContent>
      </Card>
    </div>
  );
}

function BBSummary({ summary }) {
  if (!summary) return null;

  return (
    <div className="grid grid-cols-3 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Current Price</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">${summary.current_price?.toFixed(2)}</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Band Width</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {summary.band_width_ratio?.toFixed(2)}x
          </div>
          <p className="text-xs text-gray-500">Avg ratio</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Squeeze Status</CardTitle>
        </CardHeader>
        <CardContent>
          <Badge variant={summary.squeeze_detected ? 'destructive' : 'default'}>
            {summary.squeeze_detected ? 'Squeeze Active' : 'Normal'}
          </Badge>
        </CardContent>
      </Card>
    </div>
  );
}

function SRLevels({ summary }) {
  if (!summary) return null;

  const currentPrice = summary.current_price;
  const resistance = summary.resistance_levels || [];
  const support = summary.support_levels || [];

  return (
    <div className="grid grid-cols-3 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Resistance Levels</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {resistance.length > 0 ? (
            resistance.slice(0, 3).map((level, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span>${level.toFixed(2)}</span>
                <span className="text-gray-500">
                  +{((level / currentPrice - 1) * 100).toFixed(1)}%
                </span>
              </div>
            ))
          ) : (
            <p className="text-gray-500">No resistance found</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Current Price</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">${currentPrice?.toFixed(2)}</div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Support Levels</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {support.length > 0 ? (
            support.slice(-3).reverse().map((level, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span>${level.toFixed(2)}</span>
                <span className="text-gray-500">
                  {((level / currentPrice - 1) * 100).toFixed(1)}%
                </span>
              </div>
            ))
          ) : (
            <p className="text-gray-500">No support found</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function SignalsList({ signals, title }) {
  if (!signals || signals.length === 0) {
    return <p className="text-gray-500 text-sm">No signals detected</p>;
  }

  // Show most recent 5 signals
  const recentSignals = signals.slice(-5).reverse();

  return (
    <div className="space-y-2">
      <h3 className="font-semibold text-sm">{title}</h3>
      {recentSignals.map((signal, i) => (
        <div key={i} className="border-l-4 border-blue-500 pl-3 py-2 bg-gray-50 rounded">
          <div className="flex justify-between items-start mb-1">
            <span className="font-medium text-sm capitalize">{signal.type?.replace(/_/g, ' ')}</span>
            <Badge variant="outline" className="text-xs">
              {(signal.confidence * 100).toFixed(0)}%
            </Badge>
          </div>
          <p className="text-xs text-gray-600 mb-1">{signal.evidence}</p>
          <p className="text-xs text-gray-700 italic">
            {signal.governance_narrative}
          </p>
        </div>
      ))}
    </div>
  );
}

function ReportSummary({ report }) {
  if (!report?.report_summary) return null;

  const summary = report.report_summary;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Current Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold capitalize">{summary.trend}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Momentum</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold capitalize">{summary.momentum}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Volatility</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm">{summary.volatility}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Signals</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold">{report.total_signals}</div>
          </CardContent>
        </Card>
      </div>

      {summary.key_levels && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Key Price Levels</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-2 text-sm">
              {summary.key_levels.entry && (
                <div>
                  <p className="text-gray-500">Entry</p>
                  <p className="font-bold">${summary.key_levels.entry.toFixed(2)}</p>
                </div>
              )}
              {summary.key_levels.stop_loss && (
                <div>
                  <p className="text-gray-500">Stop Loss</p>
                  <p className="font-bold text-red-600">
                    ${summary.key_levels.stop_loss.toFixed(2)}
                  </p>
                </div>
              )}
              {summary.key_levels.target_1 && (
                <div>
                  <p className="text-gray-500">Target 1</p>
                  <p className="font-bold text-green-600">
                    ${summary.key_levels.target_1.toFixed(2)}
                  </p>
                </div>
              )}
              {summary.key_levels.target_2 && (
                <div>
                  <p className="text-gray-500">Target 2</p>
                  <p className="font-bold text-green-600">
                    ${summary.key_levels.target_2.toFixed(2)}
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
