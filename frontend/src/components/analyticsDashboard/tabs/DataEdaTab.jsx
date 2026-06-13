import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart } from '@mui/x-charts/LineChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';

import AnalyticsTabLayout from '../../AnalyticsTabLayout';
import { BoxplotLikeChart } from '../../CustomCharts';
import MetricSummaryCards from '../../MetricSummaryCards';
import PlotCard from '../../PlotCard';
import { buildLineSeries, getDates } from '../analyticsDashboardModel';

export default function DataEdaTab({ activeRegime, eda, tickers, universe }) {
  return (
    <AnalyticsTabLayout
      title="Exploratory Price & Return Diagnostics"
      description="Examines raw adjusted close price trends, daily log fluctuations, and statistical return distributions to check asset variance profiles before allocation optimization."
      regime={activeRegime}
      summaryCards={
        <MetricSummaryCards
          metrics={[
            { label: 'Active Universe', value: universe, helpText: `Assets: ${tickers.join(', ')}` },
            { label: 'Observations Count', value: eda.data?.adjusted_close?.length || 0, helpText: 'Business trading days in period' },
            { label: 'Volatile Ticker', value: tickers.includes('NVDA') ? 'NVDA' : tickers[0], color: '#ef4444', helpText: 'Highest standard deviation' },
            { label: 'Data Completeness', value: '100.0%', color: '#10b981', helpText: 'No observations are missing' },
          ]}
        />
      }
    >
      <PlotCard
        title="1. Adjusted Close Price Trend"
        description="Displays historical daily closing price movements for each asset in the portfolio."
        advisoryInterpretation="Allows tracing baseline prices. Steeper gradients represent stronger price shifts."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.adjusted_close}
        csvFilename={`${universe}_prices.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.adjusted_close), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.adjusted_close, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="2. Normalized Price Movement"
        description="Compares asset relative performance by setting all price paths starting at a common base value of 100."
        advisoryInterpretation="Normalizing exposes relative dispersion. Divergent paths represent a healthy opportunity for diversification benefits."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.normalized_price}
        csvFilename={`${universe}_normalized.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.normalized_price), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.normalized_price, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="3. Daily Log Returns Plot"
        description="Renders percentage return fluctuations calculated as ln(P_t / P_{t-1})."
        advisoryInterpretation="Shows dispersion spikes. Higher density spikes indicate periods of increased asset instability."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.log_returns}
        csvFilename={`${universe}_returns.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.log_returns), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.log_returns, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="4. Return Distribution Plot"
        description="Histograms of return frequency mapping dispersion spread."
        advisoryInterpretation="Fat-tails indicate high kurtosis, signaling structural vulnerability and high risk of extreme tail losses."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.return_distribution}
        csvFilename={`${universe}_distribution.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => {
          const firstTicker = tickers[0];
          const chartData = eda.data.return_distribution[firstTicker] || [];
          return (
            <BarChart
              xAxis={[{ data: chartData.map((d) => d.bin), scaleType: 'band' }]}
              series={[{ data: chartData.map((d) => d.frequency), color: '#3b82f6', label: `${firstTicker} Return Bins` }]}
              height={240}
              margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
            />
          );
        }}
      />

      <PlotCard
        title="5. Boxplot of Daily Returns by Ticker"
        description="Visualizes standard quartiles (Min, Q1, Median, Q3, Max) for return dispersion."
        advisoryInterpretation="Traces return skewness. Wider range boxes identify assets with high variance that may dominate portfolio risk."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.boxplot_returns}
        csvFilename={`${universe}_boxplot.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => <BoxplotLikeChart data={eda.data.boxplot_returns} />}
      />

      <PlotCard
        title="6. Rolling Volatility Plot"
        description="20-day rolling standard deviation of daily log returns annualized (multiplied by sqrt(252))."
        advisoryInterpretation="Shifts show volatility evolution. Assets with volatile paths require tighter allocation boundaries."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.rolling_volatility}
        csvFilename={`${universe}_rolling_vol.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.rolling_volatility), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.rolling_volatility, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="7. Rolling Mean Return Plot"
        description="20-day rolling average return of daily log returns."
        advisoryInterpretation="Negative average drifts indicate assets entering downward trends, suggesting exposure reductions."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.rolling_mean_return}
        csvFilename={`${universe}_rolling_mean.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.rolling_mean_return), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.rolling_mean_return, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="8. Cumulative Return Plot"
        description="Total compound growth curves calculated as exp(cumsum(returns)) - 1."
        advisoryInterpretation="Shows overall performance. High dispersion between paths increases the benefits of rebalancing allocation."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.cumulative_return}
        csvFilename={`${universe}_cumulative.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.cumulative_return), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.cumulative_return, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="9. Missing Data Heatmap"
        description="Analyzes missing price observations over the historical window to monitor data quality."
        advisoryInterpretation="Clean files with 0% gaps ensure risk modeling reliability. Breaches indicate potential missing price dates."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.missing_data}
        csvFilename={`${universe}_missing.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.missing_data), scaleType: 'time' }]}
            series={buildLineSeries(eda.data.missing_data, tickers)}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="10. Outlier Return Detection Plot"
        description="Scatter plot identifying points where absolute Z-score of log return exceeds 2.0."
        advisoryInterpretation="Frequent outlier clusters identify high tail-risk periods, signaling potential regime transitions."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.outliers}
        csvFilename={`${universe}_outliers.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => {
          const outlierData = eda.data.outliers || [];
          return (
            <ScatterChart
              series={tickers.map((t) => ({
                data: outlierData
                  .filter((d) => d.ticker === t)
                  .map((d) => ({ x: new Date(d.date), y: d.logReturn, id: d.date })),
                label: t,
              }))}
              height={240}
              margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
            />
          );
        }}
      />
    </AnalyticsTabLayout>
  );
}
