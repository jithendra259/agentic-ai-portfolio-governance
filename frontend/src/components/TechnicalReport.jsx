/**
 * TechnicalReport.jsx — Detailed Technical Analysis Report Viewer
 * 
 * Displays comprehensive technical analysis report with:
 * - Market summary
 * - Technical indicators overview
 * - Governance metadata
 * - Risk assessment
 * - Actionable recommendations
 * - Signal evidence trail
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function TechnicalReport({ ticker = 'AAPL', daysBack = 252 }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `/api/technical-analysis/${ticker.toUpperCase()}/full-report?days=${daysBack}`
        );
        if (!response.ok) {
          throw new Error(`Report fetch failed: ${response.statusText}`);
        }
        const data = await response.json();
        setReport(data);
      } catch (err) {
        setError(err.message);
        console.error('Report fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    if (ticker) {
      fetchReport();
    }
  }, [ticker, daysBack]);

  if (loading) {
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold mb-4">Generating Technical Analysis Report</div>
          <div className="text-gray-500">Analyzing {ticker}...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>Failed to load report: {error}</AlertDescription>
      </Alert>
    );
  }

  if (!report) {
    return (
      <Alert>
        <AlertDescription>No report data available for {ticker}</AlertDescription>
      </Alert>
    );
  }

  const recommendation = report.recommendations?.primary_recommendation || 'NEUTRAL';
  const confidence = report.recommendations?.confidence || 0;
  const recommendationColor = {
    BUY: 'bg-green-100 text-green-800 border-green-300',
    SELL: 'bg-red-100 text-red-800 border-red-300',
    NEUTRAL: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  }[recommendation] || 'bg-gray-100 text-gray-800';

  return (
    <div className="w-full space-y-4 p-4 bg-white">
      {/* Header */}
      <div className="space-y-4">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold">{ticker} - Technical Analysis Report</h1>
            <p className="text-sm text-gray-500 mt-1">
              Generated: {new Date(report.metadata?.generated_at).toLocaleString()}
            </p>
            <p className="text-sm text-gray-500">
              Period: {report.metadata?.date_range?.start} to{' '}
              {report.metadata?.date_range?.end}
            </p>
          </div>
          <div className={`border-2 px-6 py-4 rounded-lg text-center ${recommendationColor}`}>
            <div className="text-3xl font-bold">{recommendation}</div>
            <div className="text-sm mt-1">
              Confidence: {(confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Executive Summary */}
        <Card className="border-2 border-blue-200 bg-blue-50">
          <CardHeader>
            <CardTitle>Executive Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p>{report.recommendations?.rationale}</p>
            {report.recommendations?.key_levels && (
              <div className="mt-3">
                <p className="font-semibold text-sm mb-2">Key Action Levels:</p>
                <div className="grid grid-cols-4 gap-2 text-sm">
                  {report.recommendations.key_levels.entry && (
                    <div className="bg-white p-2 rounded border">
                      <p className="text-gray-600 text-xs">Entry</p>
                      <p className="font-bold">
                        ${report.recommendations.key_levels.entry.toFixed(2)}
                      </p>
                    </div>
                  )}
                  {report.recommendations.key_levels.stop_loss && (
                    <div className="bg-white p-2 rounded border border-red-200">
                      <p className="text-gray-600 text-xs">Stop Loss</p>
                      <p className="font-bold text-red-600">
                        ${report.recommendations.key_levels.stop_loss.toFixed(2)}
                      </p>
                    </div>
                  )}
                  {report.recommendations.key_levels.target_1 && (
                    <div className="bg-white p-2 rounded border border-green-200">
                      <p className="text-gray-600 text-xs">Target 1</p>
                      <p className="font-bold text-green-600">
                        ${report.recommendations.key_levels.target_1.toFixed(2)}
                      </p>
                    </div>
                  )}
                  {report.recommendations.key_levels.target_2 && (
                    <div className="bg-white p-2 rounded border border-green-200">
                      <p className="text-gray-600 text-xs">Target 2</p>
                      <p className="font-bold text-green-600">
                        ${report.recommendations.key_levels.target_2.toFixed(2)}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Multi-Section Tabs */}
      <Tabs defaultValue="market" className="w-full">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="market">Market</TabsTrigger>
          <TabsTrigger value="trend">Trend</TabsTrigger>
          <TabsTrigger value="momentum">Momentum</TabsTrigger>
          <TabsTrigger value="volatility">Volatility</TabsTrigger>
          <TabsTrigger value="signals">Signals</TabsTrigger>
          <TabsTrigger value="governance">Governance</TabsTrigger>
        </TabsList>

        {/* MARKET SUMMARY */}
        <TabsContent value="market">
          <Card>
            <CardHeader>
              <CardTitle>Market Summary</CardTitle>
            </CardHeader>
            <CardContent>
              {report.market_summary && (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-gray-600 text-sm">Current Price</p>
                    <p className="text-3xl font-bold">
                      ${report.market_summary.current_price?.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 text-sm">Price Change (1D)</p>
                    <p
                      className={`text-2xl font-bold ${
                        report.market_summary.price_change >= 0
                          ? 'text-green-600'
                          : 'text-red-600'
                      }`}
                    >
                      {report.market_summary.price_change >= 0 ? '+' : ''}
                      ${report.market_summary.price_change?.toFixed(2)} (
                      {report.market_summary.price_change_pct?.toFixed(2)}%)
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 text-sm">Period Change</p>
                    <p
                      className={`text-2xl font-bold ${
                        report.market_summary.period_change >= 0
                          ? 'text-green-600'
                          : 'text-red-600'
                      }`}
                    >
                      {report.market_summary.period_change >= 0 ? '+' : ''}
                      ${report.market_summary.period_change?.toFixed(2)} (
                      {report.market_summary.period_change_pct?.toFixed(2)}%)
                    </p>
                  </div>

                  <div>
                    <p className="text-gray-600 text-sm">52-Week High</p>
                    <p className="text-xl font-bold">
                      ${report.market_summary['52week_high']?.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 text-sm">52-Week Low</p>
                    <p className="text-xl font-bold">
                      ${report.market_summary['52week_low']?.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 text-sm">Price Range</p>
                    <p className="text-xl font-bold">
                      ${report.market_summary.price_range?.toFixed(2)}
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* TREND ANALYSIS */}
        <TabsContent value="trend">
          <Card>
            <CardHeader>
              <CardTitle>Trend Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {report.trend_analysis && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-gray-600 text-sm">Current Trend</p>
                      <Badge className="mt-2 text-lg px-3 py-1 capitalize">
                        {report.trend_analysis.current_trend}
                      </Badge>
                      <p className="text-sm text-gray-700 mt-2">
                        Strength: {(report.trend_analysis.trend_strength * 100).toFixed(0)}%
                      </p>
                    </div>

                    <div>
                      <p className="text-gray-600 text-sm">Interpretation</p>
                      <p className="text-sm font-medium mt-2">
                        {report.trend_analysis.interpretation}
                      </p>
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="font-semibold text-sm mb-2">Moving Average Alignment:</p>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div className="bg-white p-2 rounded border">
                        <p className="text-gray-600 text-xs">SMA 20</p>
                        <p className="font-bold">
                          ${report.trend_analysis.moving_averages?.sma_20?.toFixed(2)}
                        </p>
                      </div>
                      <div className="bg-white p-2 rounded border">
                        <p className="text-gray-600 text-xs">SMA 50</p>
                        <p className="font-bold">
                          ${report.trend_analysis.moving_averages?.sma_50?.toFixed(2)}
                        </p>
                      </div>
                      <div className="bg-white p-2 rounded border">
                        <p className="text-gray-600 text-xs">SMA 200</p>
                        <p className="font-bold">
                          ${report.trend_analysis.moving_averages?.sma_200?.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  </div>

                  {report.trend_analysis.recent_signals?.length > 0 && (
                    <div>
                      <p className="font-semibold text-sm mb-2">Recent Signals:</p>
                      {report.trend_analysis.recent_signals.map((signal, i) => (
                        <div key={i} className="border-l-4 border-blue-500 pl-3 py-2 mb-2">
                          <p className="font-medium text-sm capitalize">
                            {signal.type?.replace(/_/g, ' ')}
                          </p>
                          <p className="text-xs text-gray-600">{signal.evidence}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* MOMENTUM ANALYSIS */}
        <TabsContent value="momentum">
          <Card>
            <CardHeader>
              <CardTitle>Momentum Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {report.momentum_analysis && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <p className="font-semibold mb-2">RSI (14)</p>
                      <p className="text-3xl font-bold">
                        {report.momentum_analysis.rsi?.current_value?.toFixed(1)}
                      </p>
                      <Badge className="mt-2 capitalize">
                        {report.momentum_analysis.rsi?.zone}
                      </Badge>
                      <p className="text-sm text-gray-600 mt-2">
                        {report.momentum_analysis.rsi?.interpretation}
                      </p>
                    </div>

                    <div className="bg-gray-50 p-4 rounded-lg">
                      <p className="font-semibold mb-2">MACD</p>
                      <p className="text-sm">
                        MACD Line:{' '}
                        <span className="font-bold">
                          {report.momentum_analysis.macd?.macd_line?.toFixed(4)}
                        </span>
                      </p>
                      <p className="text-sm mt-1">
                        Signal Line:{' '}
                        <span className="font-bold">
                          {report.momentum_analysis.macd?.signal_line?.toFixed(4)}
                        </span>
                      </p>
                      <p className="text-sm mt-1">
                        Histogram:{' '}
                        <span className="font-bold">
                          {report.momentum_analysis.macd?.histogram?.toFixed(4)}
                        </span>
                      </p>
                      <Badge className="mt-2 capitalize">
                        {report.momentum_analysis.macd?.momentum_direction}
                      </Badge>
                    </div>
                  </div>

                  {report.momentum_analysis.recent_signals?.length > 0 && (
                    <div>
                      <p className="font-semibold text-sm mb-2">Recent Signals:</p>
                      {report.momentum_analysis.recent_signals.map((signal, i) => (
                        <div key={i} className="border-l-4 border-orange-500 pl-3 py-2 mb-2">
                          <div className="flex justify-between items-start">
                            <p className="font-medium text-sm capitalize">
                              {signal.type?.replace(/_/g, ' ')}
                            </p>
                            <Badge variant="outline">
                              {(signal.confidence * 100).toFixed(0)}%
                            </Badge>
                          </div>
                          <p className="text-xs text-gray-600">{signal.evidence}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* VOLATILITY ANALYSIS */}
        <TabsContent value="volatility">
          <Card>
            <CardHeader>
              <CardTitle>Volatility Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {report.volatility_analysis && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <p className="font-semibold text-sm mb-2">Realized Volatility</p>
                      <p className="text-3xl font-bold">
                        {report.volatility_analysis.realized_volatility_pct?.toFixed(2)}%
                      </p>
                      <p className="text-sm text-gray-600 mt-2">
                        {report.volatility_analysis.volatility_interpretation}
                      </p>
                    </div>

                    <div className="bg-gray-50 p-4 rounded-lg">
                      <p className="font-semibold text-sm mb-2">Bollinger Bands</p>
                      <p className="text-sm">
                        Band Width:{' '}
                        <span className="font-bold">
                          ${report.volatility_analysis.bollinger_bands?.band_width?.toFixed(2)}
                        </span>
                      </p>
                      <p className="text-sm mt-1">
                        Width Ratio:{' '}
                        <span className="font-bold">
                          {report.volatility_analysis.bollinger_bands?.band_width_ratio?.toFixed(
                            2
                          )}
                          x
                        </span>
                      </p>
                      <Badge className="mt-2">
                        {report.volatility_analysis.bollinger_bands?.squeeze_detected
                          ? 'Squeeze'
                          : 'Normal'}
                      </Badge>
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="font-semibold text-sm mb-2">Interpretation</p>
                    <p className="text-sm">{report.volatility_analysis.interpretation}</p>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* SIGNALS */}
        <TabsContent value="signals">
          <div className="space-y-4">
            {/* Bullish Signals */}
            {report.signals_bullish?.length > 0 && (
              <Card className="border-l-4 border-green-500">
                <CardHeader>
                  <CardTitle className="text-green-700">Bullish Signals</CardTitle>
                  <CardDescription>
                    {report.signals_bullish.length} signals detected
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {report.signals_bullish.map((signal, i) => (
                      <div key={i} className="border rounded p-3 bg-green-50">
                        <div className="flex justify-between items-start mb-1">
                          <p className="font-medium capitalize">
                            {signal.type?.replace(/_/g, ' ')}
                          </p>
                          <Badge variant="outline" className="text-xs">
                            {(signal.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-700 mb-1">{signal.evidence}</p>
                        <p className="text-sm italic text-gray-700">
                          "{signal.governance_narrative}"
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Bearish Signals */}
            {report.signals_bearish?.length > 0 && (
              <Card className="border-l-4 border-red-500">
                <CardHeader>
                  <CardTitle className="text-red-700">Bearish Signals</CardTitle>
                  <CardDescription>
                    {report.signals_bearish.length} signals detected
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {report.signals_bearish.map((signal, i) => (
                      <div key={i} className="border rounded p-3 bg-red-50">
                        <div className="flex justify-between items-start mb-1">
                          <p className="font-medium capitalize">
                            {signal.type?.replace(/_/g, ' ')}
                          </p>
                          <Badge variant="outline" className="text-xs">
                            {(signal.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-700 mb-1">{signal.evidence}</p>
                        <p className="text-sm italic text-gray-700">
                          "{signal.governance_narrative}"
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* GOVERNANCE */}
        <TabsContent value="governance">
          <div className="space-y-4">
            {/* Risk Assessment */}
            {report.risk_assessment && (
              <Card>
                <CardHeader>
                  <CardTitle>Risk Assessment</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm text-gray-600">Risk Level</p>
                      <Badge className="mt-2" variant={
                        report.risk_assessment.risk_level === 'HIGH' ? 'destructive' :
                        report.risk_assessment.risk_level === 'MEDIUM' ? 'secondary' : 'default'
                      }>
                        {report.risk_assessment.risk_level}
                      </Badge>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm text-gray-600">Max Drawdown</p>
                      <p className="text-xl font-bold text-red-600 mt-2">
                        {report.risk_assessment.max_drawdown_pct?.toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm text-gray-600">Volatility</p>
                      <p className="text-xl font-bold mt-2">
                        {report.risk_assessment.current_volatility_pct?.toFixed(2)}%
                      </p>
                    </div>
                  </div>

                  {report.risk_assessment.risk_factors?.length > 0 && (
                    <div>
                      <p className="font-semibold text-sm mb-2">Risk Factors:</p>
                      <ul className="space-y-1">
                        {report.risk_assessment.risk_factors.map((factor, i) => (
                          <li key={i} className="text-sm text-gray-700 flex items-start">
                            <span className="mr-2">•</span>
                            {factor}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Governance Metadata */}
            {report.governance_metadata && (
              <Card>
                <CardHeader>
                  <CardTitle>Calculation & Governance Framework</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm text-gray-600">Total Signals</p>
                      <p className="text-2xl font-bold mt-2">
                        {report.governance_metadata.total_signals}
                      </p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm text-gray-600">Avg Signal Confidence</p>
                      <p className="text-2xl font-bold mt-2">
                        {(report.governance_metadata.average_signal_confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>

                  {report.governance_metadata.calculation_methods && (
                    <div>
                      <p className="font-semibold text-sm mb-2">Calculation Methods:</p>
                      <ul className="space-y-1 text-sm">
                        {Object.entries(report.governance_metadata.calculation_methods).map(
                          ([key, method], i) => (
                            <li key={i} className="text-gray-700">
                              <span className="font-medium capitalize">{key.replace(/_/g, ' ')}:</span>{' '}
                              {method}
                            </li>
                          )
                        )}
                      </ul>
                    </div>
                  )}

                  {report.governance_metadata.governance_framework && (
                    <div className="mt-4 pt-4 border-t">
                      <p className="font-semibold text-sm mb-2">Governance Framework:</p>
                      <ul className="space-y-1 text-sm">
                        <li className="text-gray-700">
                          <span className="font-medium">Evidence-Based:</span>{' '}
                          {report.governance_metadata.governance_framework.all_signals_evidence_based
                            ? '✓ All signals have explicit evidence'
                            : '✗ Some signals lack evidence'}
                        </li>
                        <li className="text-gray-700">
                          <span className="font-medium">Confidence Scoring:</span>{' '}
                          {report.governance_metadata.governance_framework.confidence_scored
                            ? '✓ All signals scored'
                            : '✗ Scoring incomplete'}
                        </li>
                        <li className="text-gray-700">
                          <span className="font-medium">Explainability:</span>{' '}
                          {report.governance_metadata.governance_framework.explainability_required
                            ? '✓ Full narratives provided'
                            : '✗ Limited explanation'}
                        </li>
                        <li className="text-gray-700">
                          <span className="font-medium">Auditability:</span>{' '}
                          {report.governance_metadata.governance_framework.auditability}
                        </li>
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="text-center text-sm text-gray-500 border-t pt-4 mt-8">
        <p>
          This report was automatically generated by the Technical Analysis System.
          <br />
          For questions about methodology or signals, consult the calculation documentation.
        </p>
      </div>
    </div>
  );
}
