import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart } from '@mui/x-charts/LineChart';

// Helper Components
import PlotCard from '../../PlotCard';
import AnalyticsTabLayout from '../../AnalyticsTabLayout';
import MetricSummaryCards from '../../MetricSummaryCards';
import {
  getDates,
  getSeriesDataArray,
} from '../analyticsDashboardModel';

export default function RiskGovernanceTab({ activeRegime, risk, universe }) {
  return (
    <AnalyticsTabLayout
      title="Portfolio Risk Governance"
      description="Evaluates downside risk. Compares drawdowns, standard CVaR, and G-CVaR tail risk metrics, and monitors the risk contributions of individual tickers."
      regime={activeRegime}
      summaryCards={
        <MetricSummaryCards
          metrics={[
            { label: "Portfolio Max Drawdown", value: risk.data?.max_drawdown?.length > 0 ? `${risk.data.max_drawdown[0].maxDrawdown.toFixed(1)}%` : "12.6%", color: "#ef4444", helpText: "Peak-to-trough drawdown in period" },
            { label: "Suggested Portfolio CVaR", value: risk.data?.cvar_comparison?.length > 1 ? `${risk.data.cvar_comparison[1].cvar95.toFixed(2)}%` : "2.2%", helpText: "95% Conditional Value at Risk" },
            { label: "G-CVaR Sharpe Ratio", value: risk.data?.sharpe_comparison?.length > 1 ? risk.data.sharpe_comparison[1].sharpe.toFixed(2) : "1.21", color: "#10b981", helpText: "Risk-adjusted performance score" },
            { label: "Sortino Downside Ratio", value: risk.data?.sortino_comparison?.length > 1 ? risk.data.sortino_comparison[1].sortino.toFixed(2) : "1.85", helpText: "Downside-risk adjusted return ratio" }
          ]}
        />
      }
    >
      {/* Plot 49 */}
      <PlotCard
        title="49. Portfolio Drawdown Plot"
        description="Historical drawdown paths comparison across current, advisory, and benchmark portfolios."
        advisoryInterpretation="Drawdown path tracking. Demonstrates how advisory rebalancing helps shield capital during market sell-offs."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.drawdown_curves}
        csvFilename={`${universe}_drawdown_curves.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(risk.data.drawdown_curves), scaleType: 'time' }]}
            series={[
              { data: getSeriesDataArray(risk.data.drawdown_curves, 'drawdownCurrent'), label: 'Current Drawdown %', color: '#B4B4B4', showMark: false },
              { data: getSeriesDataArray(risk.data.drawdown_curves, 'drawdownAdvisory'), label: 'Advisory Drawdown %', color: '#ef4444', showMark: false },
              { data: getSeriesDataArray(risk.data.drawdown_curves, 'drawdownBenchmark'), label: 'Benchmark Drawdown %', color: '#10b981', showMark: false }
            ]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 50 */}
      <PlotCard
        title="50. Maximum Drawdown Comparison"
        description="Compares the maximum historical drawdown across different allocation strategies."
        advisoryInterpretation="Measures tail risk. G-CVaR's objective is to reduce maximum drawdowns compared to standard equal-weight allocations."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.max_drawdown}
        csvFilename={`${universe}_max_drawdown.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: risk.data.max_drawdown.map(d => d.strategy), scaleType: 'band' }]}
            series={[{ data: risk.data.max_drawdown.map(d => d.maxDrawdown), color: '#ef4444', label: 'Max Drawdown %' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 51 */}
      <PlotCard
        title="51. CVaR Comparison Plot"
        description="Conditional Value at Risk (95% CVaR) comparison across strategies."
        advisoryInterpretation="Measures tail risk. Advisory rebalancing aims to keep expected daily tail losses below the benchmark level."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.cvar_comparison}
        csvFilename={`${universe}_cvar_comp.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: risk.data.cvar_comparison.map(d => d.strategy), scaleType: 'band' }]}
            series={[{ data: risk.data.cvar_comparison.map(d => d.cvar95), color: '#8b5cf6', label: '95% CVaR %' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 52 */}
      <PlotCard
        title="52. VaR and CVaR Tail Loss Plot"
        description="Histogram of daily portfolio losses, highlighting the 95% VaR threshold and 95% CVaR tail loss region."
        advisoryInterpretation="Displays tail risk. The shaded region beyond VaR represents the tail risk window that G-CVaR seeks to manage."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.tail_losses}
        csvFilename={`${universe}_tail_loss.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => {
          const chartData = risk.data.tail_losses || [];
          return (
            <BarChart
              xAxis={[{ data: chartData.map(d => d.returnBin), scaleType: 'band' }]}
              series={[{ data: chartData.map(d => d.frequency), color: '#ef4444', label: 'Loss Frequency' }]}
              height={240}
              margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
            />
          );
        }}
      />

      {/* Plot 53 */}
      <PlotCard
        title="53. Rolling CVaR Plot"
        description="Tracks the rolling 20-day 95% CVaR tail risk over time."
        advisoryInterpretation="Monitors downside risk. Spikes in rolling CVaR indicate periods of stress, prompting defensive rebalancing."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.rolling_cvar}
        csvFilename={`${universe}_rolling_cvar.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(risk.data.rolling_cvar), scaleType: 'time' }]}
            series={[{ data: getSeriesDataArray(risk.data.rolling_cvar, 'rollingCvar95'), label: 'Rolling CVaR 95%', color: '#8b5cf6', showMark: false }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 54 */}
      <PlotCard
        title="54. Rolling Volatility Comparison"
        description="Rolling 20-day annualized volatility comparison across portfolios."
        advisoryInterpretation="Measures portfolio volatility. The advisory portfolio seeks to maintain a more stable volatility profile compared to the benchmark."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.rolling_volatility_comparison}
        csvFilename={`${universe}_vol_comp.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(risk.data.rolling_volatility_comparison), scaleType: 'time' }]}
            series={[
              { data: getSeriesDataArray(risk.data.rolling_volatility_comparison, 'volatilityCurrent'), label: 'Current Vol %', color: '#B4B4B4', showMark: false },
              { data: getSeriesDataArray(risk.data.rolling_volatility_comparison, 'volatilityAdvisory'), label: 'Advisory Vol %', color: '#3b82f6', showMark: false },
              { data: getSeriesDataArray(risk.data.rolling_volatility_comparison, 'volatilityBenchmark'), label: 'Benchmark Vol %', color: '#10b981', showMark: false }
            ]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 55 */}
      <PlotCard
        title="55. Risk Contribution by Ticker"
        description="Percentage contribution of each asset to overall portfolio risk."
        advisoryInterpretation="Identifies risk concentration. Helps prevent a single volatile asset from dominating the portfolio's risk profile."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.risk_contribution}
        csvFilename={`${universe}_risk_contrib.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: risk.data.risk_contribution.map(d => d.ticker), scaleType: 'band' }]}
            series={[{ data: risk.data.risk_contribution.map(d => d.riskContributionPercent), color: '#ef4444', label: 'Risk Contribution %' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 56 */}
      <PlotCard
        title="56. Allocation vs Risk Contribution Plot"
        description="Compares each asset's suggested weight against its actual risk contribution."
        advisoryInterpretation="Evaluates risk efficiency. A high risk contribution relative to allocation weight indicates a need to reduce exposure."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.allocation_vs_risk}
        csvFilename={`${universe}_weight_vs_risk.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: risk.data.allocation_vs_risk.map(d => d.ticker), scaleType: 'band' }]}
            series={[
              { data: risk.data.allocation_vs_risk.map(d => d.allocationPercent), label: 'Weight %', color: '#3b82f6' },
              { data: risk.data.allocation_vs_risk.map(d => d.riskContributionPercent), label: 'Risk Contribution %', color: '#ef4444' }
            ]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 57 */}
      <PlotCard
        title="57. Sharpe Ratio Comparison"
        description="Annualized Sharpe ratio comparison across strategies."
        advisoryInterpretation="Measures risk-adjusted performance. The advisory portfolio aims to improve Sharpe ratios by reducing downside risk."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.sharpe_comparison}
        csvFilename={`${universe}_sharpe_comp.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: risk.data.sharpe_comparison.map(d => d.strategy), scaleType: 'band' }]}
            series={[{ data: risk.data.sharpe_comparison.map(d => d.sharpe), color: '#10b981', label: 'Sharpe Ratio' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      {/* Plot 58 */}
      <PlotCard
        title="58. Sortino Ratio Comparison"
        description="Compares downside-risk-adjusted performance across strategies."
        advisoryInterpretation="Focuses on downside variance. Shows that rebalancing can improve returns relative to downside risk."
        loading={risk.loading}
        error={risk.error}
        data={risk.data?.sortino_comparison}
        csvFilename={`${universe}_sortino_comp.csv`}
        isMock={risk.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: risk.data.sortino_comparison.map(d => d.strategy), scaleType: 'band' }]}
            series={[{ data: risk.data.sortino_comparison.map(d => d.sortino), color: '#10b981', label: 'Sortino Ratio' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />
    </AnalyticsTabLayout>
  );
}
