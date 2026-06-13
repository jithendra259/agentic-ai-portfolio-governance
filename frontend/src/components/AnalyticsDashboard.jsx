import { useState, useMemo } from 'react';
import { Box, Typography, Stack, Chip, Divider, Stepper, Step, StepLabel, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import { LineChart } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';
import { PieChart } from '@mui/x-charts/PieChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';

// Hooks
import {
  useEdaAnalytics,
  useInstabilityAnalytics,
  useAdvisoryAllocationAnalytics,
  useDiversificationAnalytics,
  useRiskGovernanceAnalytics,
  useContagionAnalytics,
  useAgentGovernanceAnalytics,
  useBacktestingAnalytics
} from '../hooks/useAnalytics';

// Helper Components
import PlotCard from './PlotCard';
import AnalyticsTabLayout from './AnalyticsTabLayout';
import MetricSummaryCards from './MetricSummaryCards';
import { HeatmapChart, NetworkGraphChart, TimelineChart, BoxplotLikeChart } from './CustomCharts';
import AnalyticsDashboardHeader from './analyticsDashboard/AnalyticsDashboardHeader';
import AnalyticsDashboardTabs from './analyticsDashboard/AnalyticsDashboardTabs';
import CorrelationCovarianceTab from './analyticsDashboard/tabs/CorrelationCovarianceTab';
import DataEdaTab from './analyticsDashboard/tabs/DataEdaTab';
import {
  buildLineSeries,
  getActiveRegime,
  getDates,
  getDateRangeForPreset,
  getSeriesDataArray,
  getUniverseTickers,
} from './analyticsDashboard/analyticsDashboardModel';

export default function AnalyticsDashboard({ setView }) {
  const [activeTab, setActiveTab] = useState(0);
  
  // Filters
  const [universe, setUniverse] = useState("U1");
  const [datePreset, setDatePreset] = useState("2024");
  const [strategy, setStrategy] = useState("G-CVaR");
  const [refreshKey, setRefreshKey] = useState(0);

  // Active tickers computed from universe selection
  const tickers = useMemo(() => getUniverseTickers(universe), [universe]);
  
  // Date range computed from preset
  const { startDate, endDate } = useMemo(() => getDateRangeForPreset(datePreset), [datePreset]);

  // Fetch grouped data for each hook
  const eda = useEdaAnalytics(tickers, startDate, endDate, refreshKey);
  const instability = useInstabilityAnalytics(tickers, startDate, endDate, refreshKey);
  const allocation = useAdvisoryAllocationAnalytics(tickers, startDate, endDate, refreshKey);
  const diversification = useDiversificationAnalytics(tickers, startDate, endDate, refreshKey);
  const risk = useRiskGovernanceAnalytics(tickers, startDate, endDate, refreshKey);
  const contagion = useContagionAnalytics(tickers, startDate, endDate, refreshKey);
  const agentGov = useAgentGovernanceAnalytics(tickers, startDate, endDate, refreshKey);
  const backtest = useBacktestingAnalytics(tickers, startDate, endDate, refreshKey);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  // Determine current active regime from timeline to display dynamically
  const activeRegime = useMemo(() => getActiveRegime(instability.data), [instability.data]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', bgcolor: '#0D0D0D', color: '#ECECEC', overflow: 'hidden' }}>
      <AnalyticsDashboardHeader
        datePreset={datePreset}
        onDatePresetChange={setDatePreset}
        onRefresh={handleRefresh}
        onStrategyChange={setStrategy}
        onUniverseChange={setUniverse}
        setView={setView}
        strategy={strategy}
        universe={universe}
      />

      <AnalyticsDashboardTabs activeTab={activeTab} onTabChange={handleTabChange} />

      {/* Dashboard View Panels */}
      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 3, bgcolor: '#0D0D0D' }}>
        
        {/* TAB 1: Data EDA */}
        {activeTab === 0 && (
          <DataEdaTab
            activeRegime={activeRegime}
            eda={eda}
            tickers={tickers}
            universe={universe}
          />
        )}
        {/* TAB 2: Correlation and Covariance EDA */}
        {activeTab === 1 && (
          <CorrelationCovarianceTab
            activeRegime={activeRegime}
            eda={eda}
            tickers={tickers}
            universe={universe}
          />
        )}
        {/* TAB 3: Instability Monitor */}
        {activeTab === 2 && (
          <AnalyticsTabLayout
            title="Composite Instability Monitor"
            description="Tracks real-time system stress. The index spikes during volatility or correlation spikes, triggering defensive adjustments and activating critical-regime rules."
            regime={activeRegime}
            summaryCards={
              <MetricSummaryCards
                metrics={[
                  { label: "Composite Index", value: instability.data?.instability_index?.length > 0 ? instability.data.instability_index[instability.data.instability_index.length-1].instabilityIndex.toFixed(2) : "0.15", color: activeRegime === "Crisis" ? "#ef4444" : "#10b981", helpText: "Combined market instability score" },
                  { label: "Trigger Threshold", value: "0.55", helpText: "Index level triggering critical-mode" },
                  { label: "Stress Score (0-100)", value: instability.data?.stress_index?.length > 0 ? `${instability.data.stress_index[instability.data.stress_index.length-1].stressScore.toFixed(0)}` : "15", helpText: "Normalized market stress index" },
                  { label: "Regime Classification", value: activeRegime, color: activeRegime === "Crisis" ? "#ef4444" : activeRegime === "Elevated" ? "#f59e0b" : "#10b981", helpText: "Active market condition classification" }
                ]}
              />
            }
          >
            {/* Plot 19 */}
            <PlotCard
              title="19. Composite Instability Index Plot"
              description="Composite Instability Index path versus the 0.55 critical trigger threshold."
              advisoryInterpretation="A breach of the threshold signals structural instability, prompting the G-CVaR optimizer to activate defensive shifts."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.instability_index}
              csvFilename={`${universe}_instability_index.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.instability_index), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(instability.data.instability_index, 'instabilityIndex'), label: 'Instability Index', color: '#ef4444', showMark: false },
                    { data: getSeriesDataArray(instability.data.instability_index, 'threshold'), label: 'Threshold Limit', color: '#f59e0b', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 20 */}
            <PlotCard
              title="20. Regime Classification Timeline"
              description="Historical band timeline showing Calm, Elevated, and Crisis regime periods."
              advisoryInterpretation="Shifts from Calm to Crisis require transitioning from standard allocations to risk-aware defensive buffers."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.regime_timeline}
              csvFilename={`${universe}_regime_timeline.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => <TimelineChart data={instability.data.regime_timeline} />}
            />

            {/* Plot 21 */}
            <PlotCard
              title="21. Market Stress Index Plot"
              description="0 to 100 scaled market stress score tracking aggregate systemic risk factors."
              advisoryInterpretation="Scores above 50 signal broad correlation clustering, indicating high likelihood of multi-asset drawdowns."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.stress_index}
              csvFilename={`${universe}_stress_index.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.stress_index), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(instability.data.stress_index, 'stressScore'), label: 'Stress Score', color: '#f59e0b', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 22 */}
            <PlotCard
              title="22. Volatility Spike Component Plot"
              description="Volatility spike component representing relative variance shifts."
              advisoryInterpretation="Volatility spikes increase marginal asset risk, prompting defensive constraint tightening."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.volatility_spike}
              csvFilename={`${universe}_vol_spike.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.volatility_spike), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(instability.data.volatility_spike, 'volatilitySpike'), label: 'Vol Spike Factor', color: '#ef4444', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 23 */}
            <PlotCard
              title="23. Correlation Spike Component Plot"
              description="Correlation spike component tracking inter-asset alignment shifts."
              advisoryInterpretation="Correlation spikes decrease diversification benefits, requiring the system to look for alternative risk-parity assets."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.correlation_spike}
              csvFilename={`${universe}_corr_spike.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.correlation_spike), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(instability.data.correlation_spike, 'correlationSpike'), label: 'Corr Spike Factor', color: '#3b82f6', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 24 */}
            <PlotCard
              title="24. Maximum Drawdown Component Plot"
              description="Drawdown component representing active cumulative peak-to-trough losses."
              advisoryInterpretation="Increasing drawdowns signal active stress, prompting the activation of capital protection rules."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.drawdown_component}
              csvFilename={`${universe}_drawdown_component.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.drawdown_component), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(instability.data.drawdown_component, 'maxDrawdownComponent'), label: 'Drawdown Component %', color: '#ef4444', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 25 */}
            <PlotCard
              title="25. Instability Component Contribution Plot"
              description="Decomposes total instability into relative contributions from volatility, correlation, and drawdowns."
              advisoryInterpretation="Pinpoints the structural driver of instability. Correlation-driven stress requires different sector diversification compared to volatility-driven spikes."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.instability_contribution}
              csvFilename={`${universe}_instability_contribution.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.instability_contribution), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(instability.data.instability_contribution, 'volatilityContribution'), label: 'Volatility Contribution', stack: 'total', showMark: false },
                    { data: getSeriesDataArray(instability.data.instability_contribution, 'correlationContribution'), label: 'Correlation Contribution', stack: 'total', showMark: false },
                    { data: getSeriesDataArray(instability.data.instability_contribution, 'drawdownContribution'), label: 'Drawdown Contribution', stack: 'total', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 26 */}
            <PlotCard
              title="26. Crisis Window Activation Plot"
              description="Traces the activation states of the crisis window based on the instability index threshold."
              advisoryInterpretation="Crisis mode activates the G-CVaR network centrality penalty term to restrict systemically risky assets."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.crisis_activation}
              csvFilename={`${universe}_crisis_activation.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(instability.data.crisis_activation), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(instability.data.crisis_activation, 'instabilityIndex'), label: 'Instability Index', color: '#ef4444', showMark: false },
                    { data: getSeriesDataArray(instability.data.crisis_activation, 'threshold'), label: 'Threshold', color: '#B4B4B4', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 27 */}
            <PlotCard
              title="27. Regime Frequency Plot"
              description="Frequency counts of Calm, Elevated, and Crisis observations in the sample."
              advisoryInterpretation="Ensures data represents all regimes. A long crisis duration warrants persistent defensive constraints."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.regime_frequency}
              csvFilename={`${universe}_regime_frequency.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: instability.data.regime_frequency.map(d => d.regime), scaleType: 'band' }]}
                  series={[{ data: instability.data.regime_frequency.map(d => d.count), color: '#10b981', label: 'Days count' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 28 */}
            <PlotCard
              title="28. Threshold Sensitivity Plot"
              description="Evaluates the impact of different instability thresholds on backtested Sharpe ratios, activation rates, and drawdowns."
              advisoryInterpretation="Calibrates the threshold. The optimal trigger minimizes drawdown spikes without causing excessive turnover drag."
              loading={instability.loading}
              error={instability.error}
              data={instability.data?.threshold_sensitivity}
              csvFilename={`${universe}_sensitivity.csv`}
              isMock={instability.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: instability.data.threshold_sensitivity.map(d => d.threshold) }]}
                  series={[
                    { data: instability.data.threshold_sensitivity.map(d => d.activationRate), label: 'Activation Rate %', color: '#f59e0b' },
                    { data: instability.data.threshold_sensitivity.map(d => d.drawdown), label: 'Drawdown Reduction %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />
          </AnalyticsTabLayout>
        )}

        {/* TAB 4: Advisory Diversification */}
        {activeTab === 3 && (
          <AnalyticsTabLayout
            title="Advisory Allocation Diversification"
            description="Examines Suggested Weight adjustments compared to current weights. Reallocates exposure across sectors, showing shifts in defensive buffer levels over time."
            regime={activeRegime}
            summaryCards={
              <MetricSummaryCards
                metrics={[
                  { label: "Cash/Defensive Buffer", value: activeRegime === "Crisis" ? "35.0%" : "5.0%", color: activeRegime === "Crisis" ? "#f59e0b" : "#10b981", helpText: "Defensive capital allocation" },
                  { label: "Suggested Tech Weight", value: "32.0%", helpText: "Total weight suggested for technology assets" },
                  { label: "Max Ticker Allocation", value: "28.0%", helpText: "Current highest suggested weight (JPM)" },
                  { label: "Average Weight Change", value: "±5.6%", helpText: "Exposure adjustments suggested" }
                ]}
              />
            }
          >
            {/* Plot 29 */}
            <PlotCard
              title="29. Current vs Advisory Allocation by Ticker"
              description="Grouped bar chart comparing current weights against advisory suggested weights."
              advisoryInterpretation="Shifts exposure. Highlights rebalancing recommendations to manage individual asset risk."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.ticker_allocation}
              csvFilename={`${universe}_weight_comparison.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.ticker_allocation.map(d => d.ticker), scaleType: 'band' }]}
                  series={[
                    { data: allocation.data.ticker_allocation.map(d => d.currentAllocation), label: 'Current Weight %', color: '#B4B4B4' },
                    { data: allocation.data.ticker_allocation.map(d => d.advisoryAllocation), label: 'Advisory Weight %', color: '#3b82f6' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 30 */}
            <PlotCard
              title="30. Current vs Advisory Sector Allocation"
              description="Compares overall GICS sector exposure before and after advisory rebalancing."
              advisoryInterpretation="Manages sector concentration, preventing excessive exposure to high-beta technology sectors during stress."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.sector_allocation}
              csvFilename={`${universe}_sector_weights.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.sector_allocation.map(d => d.sector), scaleType: 'band' }]}
                  series={[
                    { data: allocation.data.sector_allocation.map(d => d.currentAllocation), label: 'Current Weight %', color: '#B4B4B4' },
                    { data: allocation.data.sector_allocation.map(d => d.advisoryAllocation), label: 'Advisory Weight %', color: '#10b981' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 31 */}
            <PlotCard
              title="31. Advisory Allocation Pie Chart"
              description="Visualizes the advisory portfolio composition as a donut chart."
              advisoryInterpretation="Displays target allocation weights, highlighting the recommended distribution of capital."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.advisory_pie}
              csvFilename={`${universe}_pie_weights.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <PieChart
                  series={[{
                    data: allocation.data.advisory_pie.map(d => ({ id: d.id, value: d.value, label: d.id })),
                    innerRadius: 40,
                    outerRadius: 80,
                    paddingAngle: 2,
                    cornerRadius: 3
                  }]}
                  height={240}
                />
              )}
            />

            {/* Plot 32 */}
            <PlotCard
              title="32. Allocation Change by Ticker"
              description="Displays net advisory exposure adjustments in percentage points."
              advisoryInterpretation="Highlights the direction of recommended changes. Negative values indicate assets where risk constraints suggest reducing exposure."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.allocation_change}
              csvFilename={`${universe}_weight_changes.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.allocation_change.map(d => d.ticker), scaleType: 'band' }]}
                  series={[{ data: allocation.data.allocation_change.map(d => d.allocationChange), label: 'Weight Change (pps)', color: '#ef4444' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 33 */}
            <PlotCard
              title="33. Allocation Adaptation Over Time"
              description="Traces how G-CVaR dynamically adjusts advisory weights over the backtest window."
              advisoryInterpretation="Highlights risk-responsive allocation. Advisory weights adapt dynamically, shifting from volatile assets to defensive buffers during stress."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.allocation_adaptation}
              csvFilename={`${universe}_weight_timeline.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(allocation.data.allocation_adaptation), scaleType: 'time' }]}
                  series={buildLineSeries(allocation.data.allocation_adaptation, tickers)}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 34 */}
            <PlotCard
              title="34. Critical Condition Allocation Shift"
              description="Compares advisory weights under normal (Calm) and critical (Crisis) regimes."
              advisoryInterpretation="Visualizes the stress-response policy. Demonstrates the shifting of tech exposure into defensive assets during periods of high instability."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.critical_shift}
              csvFilename={`${universe}_crisis_weights.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.critical_shift.map(d => d.ticker), scaleType: 'band' }]}
                  series={[
                    { data: allocation.data.critical_shift.map(d => d.normalAllocation), label: 'Normal Weight %', color: '#10b981' },
                    { data: allocation.data.critical_shift.map(d => d.criticalAllocation), label: 'Critical Weight %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 35 */}
            <PlotCard
              title="35. Ticker Exposure Waterfall"
              description="Step-wise presentation of recommended exposure adjustments starting from equal weights."
              advisoryInterpretation="Explains the allocation adjustments. Explains how the portfolio transitions towards the target risk-managed state."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.waterfall}
              csvFilename={`${universe}_waterfall.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.waterfall.map(d => d.ticker), scaleType: 'band' }]}
                  series={[{ data: allocation.data.waterfall.map(d => d.allocationChange), label: 'Rebalancing Change', color: '#8b5cf6' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 36 */}
            <PlotCard
              title="36. Cash/Defensive Buffer Plot"
              description="Tracks the recommended defensive cash buffer allocation across market regimes."
              advisoryInterpretation="A higher buffer during crisis periods helps shield portfolio value from severe systemic drawdowns."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.cash_buffer}
              csvFilename={`${universe}_cash_buffer.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(allocation.data.cash_buffer), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(allocation.data.cash_buffer, 'cashAllocation'), label: 'Cash Buffer %', color: '#f59e0b', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 37 */}
            <PlotCard
              title="37. Allocation Constraint Boundary Plot"
              description="Evaluates asset weights against the 30% concentration limits defined by governance rules."
              advisoryInterpretation="Identifies threshold compliance. The system automatically trims weights that approach boundary limits to prevent concentration risk."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.constraints}
              csvFilename={`${universe}_constraints.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.constraints.map(d => d.ticker), scaleType: 'band' }]}
                  series={[
                    { data: allocation.data.constraints.map(d => d.advisoryAllocation), label: 'Advisory Weight %', color: '#3b82f6' },
                    { data: allocation.data.constraints.map(d => d.maxAllowed), label: 'Max Constraint Limit %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 38 */}
            <PlotCard
              title="38. Before vs After Diversification Map"
              description="Visualizes asset and sector exposure adjustments across GICS sectors."
              advisoryInterpretation="Displays sector-level rebalancing. Demonstrates the shifting of exposure towards a more diversified sector structure."
              loading={allocation.loading}
              error={allocation.error}
              data={allocation.data?.diversification_map}
              csvFilename={`${universe}_diversification_map.csv`}
              isMock={allocation.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: allocation.data.diversification_map.map(d => d.ticker), scaleType: 'band' }]}
                  series={[
                    { data: allocation.data.diversification_map.map(d => d.currentAllocation), label: 'Before Weight %', color: '#B4B4B4' },
                    { data: allocation.data.diversification_map.map(d => d.advisoryAllocation), label: 'After Weight %', color: '#10b981' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />
          </AnalyticsTabLayout>
        )}

        {/* TAB 5: Diversification Diagnostics */}
        {activeTab === 4 && (
          <AnalyticsTabLayout
            title="Concentration & Diversification Diagnostics"
            description="Evaluates portfolio concentration using the Herfindahl-Hirschman Index (HHI), effective number of holdings, and diversification ratio over the backtest window."
            regime={activeRegime}
            summaryCards={
              <MetricSummaryCards
                metrics={[
                  { label: "Advisory HHI Index", value: diversification.data?.hhi_index?.length > 0 ? diversification.data.hhi_index[diversification.data.hhi_index.length-1].hhiAdvisory.toFixed(3) : "0.224", color: "#10b981", helpText: "HHI score (lower is more diversified)" },
                  { label: "Effective Holdings count", value: diversification.data?.effective_holdings?.length > 0 ? diversification.data.effective_holdings[diversification.data.effective_holdings.length-1].effectiveNAdvisory.toFixed(1) : "4.5", helpText: "Equivalent number of equal-weight holdings" },
                  { label: "Diversification Score", value: "84.2 / 100", color: "#10b981", helpText: "Composite diversification score" },
                  { label: "Active Breaches", value: "2", color: "#ef4444", helpText: "Limit breaches needing review" }
                ]}
              />
            }
          >
            {/* Plot 39 */}
            <PlotCard
              title="39. Herfindahl-Hirschman Index Plot"
              description="HHI concentration scores (sum of squared asset weights) over time."
              advisoryInterpretation="HHI values below 0.15 indicate low concentration, values above 0.25 signal elevated asset concentration risk."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.hhi_index}
              csvFilename={`${universe}_hhi.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(diversification.data.hhi_index), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(diversification.data.hhi_index, 'hhiCurrent'), label: 'Current HHI', color: '#B4B4B4', showMark: false },
                    { data: getSeriesDataArray(diversification.data.hhi_index, 'hhiAdvisory'), label: 'Advisory HHI', color: '#3b82f6', showMark: false },
                    { data: getSeriesDataArray(diversification.data.hhi_index, 'hhiBenchmark'), label: 'Benchmark HHI', color: '#10b981', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 40 */}
            <PlotCard
              title="40. Effective Number of Holdings Plot"
              description="Tracks the effective number of diversified assets, calculated as 1/HHI."
              advisoryInterpretation="G-CVaR maintains the effective asset count close to the total asset count, avoiding risk concentration in a single dominant asset."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.effective_holdings}
              csvFilename={`${universe}_effective_assets.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(diversification.data.effective_holdings), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(diversification.data.effective_holdings, 'effectiveNCurrent'), label: 'Current N_eff', color: '#B4B4B4', showMark: false },
                    { data: getSeriesDataArray(diversification.data.effective_holdings, 'effectiveNAdvisory'), label: 'Advisory N_eff', color: '#3b82f6', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 41 */}
            <PlotCard
              title="41. Diversification Score Before vs After"
              description="Compares the portfolio's composite diversification score before and after advisory rebalancing."
              advisoryInterpretation="A higher score indicates improved diversification, reflecting a more balanced distribution of risk across sectors."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.diversification_score}
              csvFilename={`${universe}_scores.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: diversification.data.diversification_score.map(d => d.portfolioVersion), scaleType: 'band' }]}
                  series={[{ data: diversification.data.diversification_score.map(d => d.diversificationScore), color: '#10b981', label: 'Diversification Score' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 42 */}
            <PlotCard
              title="42. Ticker Concentration Plot"
              description="Individual asset concentration levels against the 25% single-name limit."
              advisoryInterpretation="Maintains single-asset concentration limits. Highlights suggested exposure reductions if weights approach the limit."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.ticker_concentration}
              csvFilename={`${universe}_ticker_concentration.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: diversification.data.ticker_concentration.map(d => d.ticker), scaleType: 'band' }]}
                  series={[
                    { data: diversification.data.ticker_concentration.map(d => d.allocation), label: 'Advisory Weight %', color: '#3b82f6' },
                    { data: diversification.data.ticker_concentration.map(d => d.threshold), label: 'Governance Limit %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 43 */}
            <PlotCard
              title="43. Sector Concentration Plot"
              description="Sector exposure levels against the 50% concentration limit."
              advisoryInterpretation="A sector weight above the 50% limit trigger prompts a notification to restructure sector allocations."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.sector_concentration}
              csvFilename={`${universe}_sector_concentration.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: diversification.data.sector_concentration.map(d => d.sector), scaleType: 'band' }]}
                  series={[
                    { data: diversification.data.sector_concentration.map(d => d.allocation), label: 'Sector Weight %', color: '#10b981' },
                    { data: diversification.data.sector_concentration.map(d => d.threshold), label: 'Governance Limit %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 44 */}
            <PlotCard
              title="44. Concentration Threshold Breach Plot"
              description="Identifies active concentration limit breaches across assets and sectors."
              advisoryInterpretation="Highlights breaches needing attention. Triggers warnings if weights exceed the maximum allowed limits."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.breaches}
              csvFilename={`${universe}_breaches.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent', boxShadow: 'none' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Name</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Type</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Weight</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Limit</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {diversification.data.breaches.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.name}</TableCell>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.type}</TableCell>
                          <TableCell align="right" sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.allocation}%</TableCell>
                          <TableCell align="right" sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.threshold}%</TableCell>
                          <TableCell align="right" sx={{ py: 0.5 }}>
                            <Chip
                              label={row.status}
                              size="small"
                              sx={{
                                height: 20,
                                fontSize: '0.65rem',
                                fontWeight: 700,
                                bgcolor: row.status === 'BREACH' ? 'rgba(239, 68, 68, 0.08)' : 'rgba(16, 185, 129, 0.08)',
                                color: row.status === 'BREACH' ? '#ef4444' : '#10b981',
                                border: `1px solid ${row.status === 'BREACH' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`
                              }}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            />

            {/* Plot 45 */}
            <PlotCard
              title="45. Diversification Ratio Plot"
              description="Tracks the portfolio's diversification ratio (weighted volatility divided by portfolio volatility) over time."
              advisoryInterpretation="A higher ratio indicates more effective diversification, showing that portfolio volatility is lower than the weighted average of its components."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.diversification_ratio}
              csvFilename={`${universe}_div_ratio.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(diversification.data.diversification_ratio), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(diversification.data.diversification_ratio, 'diversificationRatio'), label: 'Ratio Value', color: '#10b981', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 46 */}
            <PlotCard
              title="46. Top Holdings Concentration Plot"
              description="Cumulative weight of the Top 1, Top 3, and Top 5 holdings."
              advisoryInterpretation="Highlights index concentration. Reducing the cumulative weight of top holdings helps mitigate single-name risk."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.top_holdings}
              csvFilename={`${universe}_top_holdings.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: diversification.data.top_holdings.map(d => d.bucket), scaleType: 'band' }]}
                  series={[{ data: diversification.data.top_holdings.map(d => d.exposurePercent), color: '#8b5cf6', label: 'Cumulative Weight %' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 47 */}
            <PlotCard
              title="47. Portfolio Weight Dispersion Plot"
              description="Quartiles of asset weight dispersion under current and advisory allocations."
              advisoryInterpretation="Shows the distribution of asset weights. Rebalancing shifts the distribution from equal weights to a broader, risk-managed spread."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.weight_dispersion}
              csvFilename={`${universe}_dispersion.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => <BoxplotLikeChart data={diversification.data.weight_dispersion} />}
            />

            {/* Plot 48 */}
            <PlotCard
              title="48. Equal Weight Distance Plot"
              description="Displays weight deviations of each asset from an equal-weight baseline (20%)."
              advisoryInterpretation="Identifies deviations from equal-weight. Focuses exposure on lower-risk assets while trimming those with higher systemic risk profiles."
              loading={diversification.loading}
              error={diversification.error}
              data={diversification.data?.distance_equal}
              csvFilename={`${universe}_distance.csv`}
              isMock={diversification.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: diversification.data.distance_equal.map(d => d.ticker), scaleType: 'band' }]}
                  series={[{ data: diversification.data.distance_equal.map(d => d.distanceFromEqualWeight), label: 'Distance from 20% (pps)', color: '#3b82f6' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />
          </AnalyticsTabLayout>
        )}

        {/* TAB 6: Risk Governance */}
        {activeTab === 5 && (
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
        )}

        {/* TAB 7: Contagion Graph Analysis */}
        {activeTab === 6 && (
          <AnalyticsTabLayout
            title="Institutional Contagion Graph Analysis"
            description="Examines systemic risk linkages. Uses institutional co-ownership centrality to apply risk penalties and adjust allocations away from vulnerable contagion nodes."
            regime={activeRegime}
            summaryCards={
              <MetricSummaryCards
                metrics={[
                  { label: "Overlapping Owner Density", value: contagion.data?.co_ownership_density?.length > 0 ? `${(contagion.data.co_ownership_density[contagion.data.co_ownership_density.length-1].coOwnershipDensity * 100).toFixed(1)}%` : "42.0%", helpText: "Asset co-ownership overlap density" },
                  { label: "Top Central Ticker", value: contagion.data?.eigenvector_centrality?.length > 1 ? contagion.data.eigenvector_centrality[1].ticker : "MSFT", color: "#ef4444", helpText: "Stock with highest network centrality" },
                  { label: "Sigmoid Trust Lambda", value: contagion.data?.graph_penalty?.length > 0 ? contagion.data.graph_penalty[contagion.data.graph_penalty.length-1].lambda.toFixed(2) : "0.15", helpText: "Active network penalty multiplier" },
                  { label: "Top Institutional Holder", value: "Vanguard Group", helpText: "Holds average 18.4% share across assets" }
                ]}
              />
            }
          >
            {/* Plot 59 */}
            <PlotCard
              title="59. Institution-Asset Bipartite Graph"
              description="Bipartite graph mapping institutional holdings connections to portfolio assets."
              advisoryInterpretation="Identifies common institutional owners, helping highlight potential liquidation paths during systemic sell-offs."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.nodes}
              csvFilename={`${universe}_bipartite_nodes.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => <NetworkGraphChart nodes={contagion.data.nodes} edges={contagion.data.edges} label="Institutional Holdings Connections" />}
            />

            {/* Plot 60 */}
            <PlotCard
              title="60. Ticker Co-Ownership Graph"
              description="Asset co-ownership network mapping, where edges represent common institutional ownership."
              advisoryInterpretation="Stronger edges indicate higher co-ownership, representing potential channels for cross-asset contagion."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.co_nodes}
              csvFilename={`${universe}_coownership_edges.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => <NetworkGraphChart nodes={contagion.data.co_nodes} edges={contagion.data.co_edges} label="Asset Co-Ownership Network" />}
            />

            {/* Plot 61 */}
            <PlotCard
              title="61. Eigenvector Centrality by Ticker"
              description="Tracks asset eigenvector centrality within the co-ownership network."
              advisoryInterpretation="Higher centrality indicates a more systemically connected asset, which may require tighter exposure constraints."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.eigenvector_centrality}
              csvFilename={`${universe}_centrality.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: contagion.data.eigenvector_centrality.map(d => d.ticker), scaleType: 'band' }]}
                  series={[{ data: contagion.data.eigenvector_centrality.map(d => d.eigenvectorCentrality), color: '#ef4444', label: 'Eigenvector Centrality' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 62 */}
            <PlotCard
              title="62. Contagion Penalty Score Plot"
              description="Calculates risk penalty scores based on co-ownership centrality and asset volatility."
              advisoryInterpretation="Higher penalty scores indicate assets that may experience larger allocation reductions under stress."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.contagion_penalty}
              csvFilename={`${universe}_penalty.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: contagion.data.contagion_penalty.map(d => d.ticker), scaleType: 'band' }]}
                  series={[{ data: contagion.data.contagion_penalty.map(d => d.penaltyScore), color: '#8b5cf6', label: 'Contagion Risk Penalty' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 63 */}
            <PlotCard
              title="63. Centrality vs Advisory Weight Plot"
              description="Scatter plot comparing asset centrality against its advisory weight."
              advisoryInterpretation="Evaluates risk-adjusted weights. Systemically central assets generally receive lower weights during stress to manage contagion risk."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.centrality_vs_weight}
              csvFilename={`${universe}_centrality_vs_weight.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => {
                const scatterData = contagion.data.centrality_vs_weight || [];
                return (
                  <ScatterChart
                    series={[{
                      data: scatterData.map(d => ({ x: d.eigenvectorCentrality, y: d.advisoryWeight, id: d.ticker })),
                      label: 'Centrality vs Advisory Weight'
                    }]}
                    height={240}
                    margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                  />
                );
              }}
            />

            {/* Plot 64 */}
            <PlotCard
              title="64. Centrality vs Weight Change Plot"
              description="Scatter plot comparing asset centrality against its suggested exposure change."
              advisoryInterpretation="Highlights risk management. Assets with high centrality tend to see weight reductions during market stress."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.centrality_vs_change}
              csvFilename={`${universe}_centrality_vs_change.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => {
                const scatterData = contagion.data.centrality_vs_change || [];
                return (
                  <ScatterChart
                    series={[{
                      data: scatterData.map(d => ({ x: d.eigenvectorCentrality, y: d.allocationChange, id: d.ticker })),
                      label: 'Centrality vs Weight Change'
                    }]}
                    height={240}
                    margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                  />
                );
              }}
            />

            {/* Plot 65 */}
            <PlotCard
              title="65. Graph-Regularized CVaR Penalty Plot"
              description="Tracks the graph penalty term (lambda_t * Penalty) over the backtest window."
              advisoryInterpretation="Monitors penalty strength. The penalty term increases during crisis periods, prioritizing capital protection."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.graph_penalty}
              csvFilename={`${universe}_rolling_penalty.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(contagion.data.graph_penalty), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(contagion.data.graph_penalty, 'graphPenalty'), label: 'Active Penalty Score', color: '#ef4444', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 66 */}
            <PlotCard
              title="66. Sigmoid Trust Function Plot"
              description="Visualizes how the penalty multiplier lambda scales with the Composite Instability Index."
              advisoryInterpretation="Displays the penalty curve. The multiplier increases as instability approaches the trigger threshold."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.sigmoid_curve}
              csvFilename={`${universe}_sigmoid.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: contagion.data.sigmoid_curve.map(d => d.instabilityIndex) }]}
                  series={[{ data: contagion.data.sigmoid_curve.map(d => d.lambda), label: 'Lambda Multiplier', color: '#f59e0b' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 67 */}
            <PlotCard
              title="67. Co-Ownership Density Plot"
              description="Tracks overall co-ownership network density over time."
              advisoryInterpretation="Monitors market concentration. Rising density indicates increasing co-ownership, which can amplify contagion risk."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.co_ownership_density}
              csvFilename={`${universe}_density.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(contagion.data.co_ownership_density), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(contagion.data.co_ownership_density, 'coOwnershipDensity'), label: 'Network Density', color: '#10b981', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 68 */}
            <PlotCard
              title="68. Top Institutional Holder Exposure Plot"
              description="Displays average holding shares for the top institutional managers across the portfolio."
              advisoryInterpretation="Identifies key institutional holders, helping monitor potential capital flow risks from major asset managers."
              loading={contagion.loading}
              error={contagion.error}
              data={contagion.data?.top_holders_exposure}
              csvFilename={`${universe}_holders.csv`}
              isMock={contagion.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: contagion.data.top_holders_exposure.map(d => d.institution), scaleType: 'band' }]}
                  series={[{ data: contagion.data.top_holders_exposure.map(d => d.exposurePercent), color: '#10b981', label: 'Average Share %' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />
          </AnalyticsTabLayout>
        )}

        {/* TAB 8: Agent Governance and Explainability */}
        {activeTab === 7 && (
          <AnalyticsTabLayout
            title="Multi-Agent Blackboard & HITL Audit Trail"
            description="Monitors the blackboard audit trail. Displays the status of the five-agent pipeline, logs trigger reasons, and reviews human-in-the-loop decisions."
            regime={activeRegime}
            summaryCards={
              <MetricSummaryCards
                metrics={[
                  { label: "Active Pipelines", value: "A0 to A4 Active", color: "#10b981", helpText: "All 5 specialized agents operational" },
                  { label: "Traceability Claims", value: "3 Claims Audited", helpText: "Blackboard numeric values verified" },
                  { label: "Admin Decisions", value: "2 Logged Decisions", helpText: "Human accept/constrain actions" },
                  { label: "compliance Rate", value: "75.0%", color: "#f59e0b", helpText: "Rules passing governance checks" }
                ]}
              />
            }
          >
            {/* Plot 69 */}
            <PlotCard
              title="69. Five-Agent Pipeline Status"
              description="Stepper diagram showing the status of the five-agent pipeline."
              advisoryInterpretation="Confirms process completion, ensuring that data validation, risk modeling, and explanation generation are executed."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.pipeline_status}
              csvFilename={`${universe}_pipeline_status.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <Stepper activeStep={5} orientation="vertical" sx={{ py: 1 }}>
                  {agentGov.data.pipeline_status.map((step, index) => (
                    <Step key={index} completed={true}>
                      <StepLabel>
                        <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.primary' }}>
                          {step.agentName} ({step.status})
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.2 }}>
                          {step.outputSummary} [Duration: {step.startedAt} to {step.completedAt}]
                        </Typography>
                      </StepLabel>
                    </Step>
                  ))}
                </Stepper>
              )}
            />

            {/* Plot 70 */}
            <PlotCard
              title="70. Blackboard Audit Trail Timeline"
              description="Chronological log of agent actions and updates recorded on the shared blackboard."
              advisoryInterpretation="Ensures process auditability, allowing tracing of allocations back to the specific agent inputs."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.audit_trail}
              csvFilename={`${universe}_audit_trail.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent', boxShadow: 'none', maxHeight: '200px' }}>
                  <Table size="small">
                    <TableBody>
                      {agentGov.data.audit_trail.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell sx={{ color: 'text.secondary', fontSize: '0.68rem', py: 0.5 }}>{row.timestamp}</TableCell>
                          <TableCell sx={{ color: 'text.primary', fontWeight: 600, fontSize: '0.72rem', py: 0.5 }}>{row.agentName}</TableCell>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.72rem', py: 0.5 }}>{row.action}</TableCell>
                          <TableCell sx={{ color: 'text.secondary', fontSize: '0.68rem', py: 0.5 }}>[{row.blackboardCollection}]</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            />

            {/* Plot 71 */}
            <PlotCard
              title="71. HITL Trigger Timeline"
              description="Timeline tracking events that triggered human-in-the-loop (HITL) review requests."
              advisoryInterpretation="Identifies review requests. Triggers occur during regime shifts or when suggested turnover exceeds the 15% limit."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.hitl_triggers}
              csvFilename={`${universe}_hitl_triggers.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent', boxShadow: 'none' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Date</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Trigger Reason</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Instability</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Turnover</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {agentGov.data.hitl_triggers.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.date}</TableCell>
                          <TableCell sx={{ color: 'text.primary', fontWeight: 600, fontSize: '0.75rem' }}>{row.triggerType}</TableCell>
                          <TableCell align="right" sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.instabilityIndex}</TableCell>
                          <TableCell align="right" sx={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.75rem' }}>{row.turnover}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            />

            {/* Plot 72 */}
            <PlotCard
              title="72. Governance Decision Log"
              description="Historical log of actions taken by administrators in response to review requests."
              advisoryInterpretation="Logs administrative overrides, documenting approved adjustments and specific constraints applied to the portfolio."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.decision_log}
              csvFilename={`${universe}_governance_decisions.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent', boxShadow: 'none' }}>
                  <Table size="small">
                    <TableBody>
                      {agentGov.data.decision_log.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell sx={{ py: 1 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{row.timestamp} | Window: {row.windowId}</Typography>
                            <Typography variant="subtitle2" sx={{ fontWeight: 800, color: row.action === 'ACCEPT' ? '#10b981' : '#f59e0b', mt: 0.2 }}>
                              {row.action}: {row.reason}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.2 }}>
                              Prev: {row.previousWeightSummary} | Final: {row.finalWeightSummary}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            />

            {/* Plot 73 */}
            <PlotCard
              title="73. Agent Explanation Panel"
              description="Generated narrative explaining allocation adjustments and highlighting top risk drivers."
              advisoryInterpretation="Provides human-readable explanations, detailing the rationale behind rebalancing recommendations."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.explanation}
              csvFilename={`${universe}_explanations.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <Stack spacing={2} sx={{ py: 1 }}>
                  <Typography variant="body2" sx={{ lineHeight: 1.5, color: 'text.primary' }}>
                    {agentGov.data.explanation.narrative}
                  </Typography>
                  <Divider sx={{ bgcolor: '#262626' }} />
                  <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary', textTransform: 'uppercase' }}>
                    Top Risk Drivers
                  </Typography>
                  {agentGov.data.explanation.topRiskDrivers.map((driver, idx) => (
                    <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                      <span style={{ color: '#e5e7eb' }}>{driver.driver}</span>
                      <span style={{ color: '#f59e0b', fontWeight: 600 }}>{driver.impact}</span>
                    </Box>
                  ))}
                </Stack>
              )}
            />

            {/* Plot 74 */}
            <PlotCard
              title="74. Numerical Claim Traceability Table"
              description="Traceability table mapping narrative claims to underlying blackboard metrics."
              advisoryInterpretation="Ensures explanation accuracy, linking narrative statements back to verifiable calculation values."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.traceability}
              csvFilename={`${universe}_traceability.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent', boxShadow: 'none' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Claim Statement</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Value</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Source Collection</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {agentGov.data.traceability.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.claim}</TableCell>
                          <TableCell sx={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.75rem' }}>{row.value}</TableCell>
                          <TableCell sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>{row.sourceCollection}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            />

            {/* Plot 75 */}
            <PlotCard
              title="75. Trigger Reason Breakdown"
              description="Distribution of trigger reasons for human-in-the-loop review requests."
              advisoryInterpretation="Monitors governance triggers. A high frequency of regime-shift triggers indicates persistent market instability."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.trigger_reasons}
              csvFilename={`${universe}_trigger_reasons.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: agentGov.data.trigger_reasons.map(d => d.reason), scaleType: 'band' }]}
                  series={[{ data: agentGov.data.trigger_reasons.map(d => d.count), color: '#8b5cf6', label: 'Trigger Count' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 76 */}
            <PlotCard
              title="76. Before-HITL vs After-HITL Allocation"
              description="Compares suggested weights before and after human-in-the-loop overrides."
              advisoryInterpretation="Visualizes administrative overrides. Shows adjustments made to target weights based on manual review."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.before_after_hitl}
              csvFilename={`${universe}_hitl_overrides.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: agentGov.data.before_after_hitl.map(d => d.ticker), scaleType: 'band' }]}
                  series={[
                    { data: agentGov.data.before_after_hitl.map(d => d.beforeHitlAllocation), label: 'G-CVaR Base Weight %', color: '#B4B4B4' },
                    { data: agentGov.data.before_after_hitl.map(d => d.afterHitlAllocation), label: 'HITL Approved Weight %', color: '#10b981' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 77 */}
            <PlotCard
              title="77. Turnover Alert Plot"
              description="Suggested portfolio turnover relative to the 15% governance threshold."
              advisoryInterpretation="Monitors portfolio turnover. Turnover spikes suggest significant weight adjustments, which may require administrative confirmation."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.turnover_alerts}
              csvFilename={`${universe}_turnover_alerts.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(agentGov.data.turnover_alerts), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(agentGov.data.turnover_alerts, 'turnover'), label: 'Turnover %', color: '#f59e0b', showMark: false },
                    { data: getSeriesDataArray(agentGov.data.turnover_alerts, 'turnoverThreshold'), label: 'Limit Threshold', color: '#ef4444', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 78 */}
            <PlotCard
              title="78. Governance Rule Compliance Matrix"
              description="Compliance status of the portfolio relative to key governance limits."
              advisoryInterpretation="Traces compliance status. Rule compliance is monitored automatically, triggering alerts if limit breaches occur."
              loading={agentGov.loading}
              error={agentGov.error}
              data={agentGov.data?.compliance_matrix}
              csvFilename={`${universe}_compliance.csv`}
              isMock={agentGov.data?.is_mock}
              renderChart={() => (
                <TableContainer component={Paper} sx={{ bgcolor: 'transparent', boxShadow: 'none' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Rule Name</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Value</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Limit</TableCell>
                        <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {agentGov.data.compliance_matrix.map((row, idx) => (
                        <TableRow key={idx}>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.ruleName}</TableCell>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.currentValue}</TableCell>
                          <TableCell sx={{ color: 'text.primary', fontSize: '0.75rem' }}>{row.threshold}</TableCell>
                          <TableCell align="right" sx={{ py: 0.5 }}>
                            <Chip
                              label={row.status}
                              size="small"
                              sx={{
                                height: 20,
                                fontSize: '0.65rem',
                                fontWeight: 700,
                                bgcolor: row.status === 'FAIL' ? 'rgba(239, 68, 68, 0.08)' : 'rgba(16, 185, 129, 0.08)',
                                color: row.status === 'FAIL' ? '#ef4444' : '#10b981',
                                border: `1px solid ${row.status === 'FAIL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`
                              }}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            />
          </AnalyticsTabLayout>
        )}

        {/* TAB 9: Evaluation and Backtesting */}
        {activeTab === 8 && (
          <AnalyticsTabLayout
            title="Historical Backtesting & Strategy Evaluation"
            description="Evaluates advisory portfolio performance over the backtest window. Compares growth paths, transaction cost impacts, and ablation metrics."
            regime={activeRegime}
            summaryCards={
              <MetricSummaryCards
                metrics={[
                  { label: "Advisory Annual Return", value: "14.8%", color: "#10b981", helpText: "Annual return in backtest" },
                  { label: "Equal Weight return", value: "12.4%", helpText: "Baseline rebalanced return" },
                  { label: "Transaction Slippage Drag", value: "0.15%", helpText: "Estimated cost drag on returns" },
                  { label: "Validation Robustness", value: "Passed", color: "#10b981", helpText: "IS/OOS Sharpe checks compliant" }
                ]}
              />
            }
          >
            {/* Plot 79 */}
            <PlotCard
              title="79. Advisory Portfolio vs Equal Weight Equity Curve"
              description="Historical equity curve comparison starting from a base value of 10,000."
              advisoryInterpretation="Tracks cumulative growth. Demonstrates the performance of the advisory portfolio relative to an equal-weight baseline."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.equity_curves}
              csvFilename={`${universe}_backtest_ew.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(backtest.data.equity_curves), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(backtest.data.equity_curves, 'advisoryPortfolioValue'), label: 'Advisory G-CVaR', color: '#f59e0b', showMark: false },
                    { data: getSeriesDataArray(backtest.data.equity_curves, 'equalWeightValue'), label: 'Equal Weight (EW)', color: '#B4B4B4', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 50, right: 10 }}
                />
              )}
            />

            {/* Plot 80 */}
            <PlotCard
              title="80. Advisory Portfolio vs Standard CVaR Equity Curve"
              description="Equity curve comparison between the graph-regularized advisory portfolio and standard CVaR."
              advisoryInterpretation="Highlights model comparisons. Demonstrates the impact of including network centrality constraints."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.equity_curves}
              csvFilename={`${universe}_backtest_std.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(backtest.data.equity_curves), scaleType: 'time' }]}
                  series={[
                    { data: getSeriesDataArray(backtest.data.equity_curves, 'advisoryPortfolioValue'), label: 'Advisory G-CVaR', color: '#f59e0b', showMark: false },
                    { data: getSeriesDataArray(backtest.data.equity_curves, 'standardCvarValue'), label: 'Standard CVaR', color: '#8b5cf6', showMark: false }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 50, right: 10 }}
                />
              )}
            />

            {/* Plot 81 */}
            <PlotCard
              title="81. Strategy Performance Comparison"
              description="Compares annualized return, volatility, Sharpe, CVaR, and max drawdown across strategies."
              advisoryInterpretation="Evaluates strategy performance, highlighting the impact of different allocation models on key risk-return metrics."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.performance}
              csvFilename={`${universe}_performance_comp.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.performance.map(d => d.strategy), scaleType: 'band' }]}
                  series={[
                    { data: backtest.data.performance.map(d => d.annualReturn), label: 'Annual Return %', color: '#10b981' },
                    { data: backtest.data.performance.map(d => d.maxDrawdown), label: 'Max Drawdown %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 82 */}
            <PlotCard
              title="82. Crisis-Regime Drawdown Comparison"
              description="Maximum drawdowns during identified crisis regimes."
              advisoryInterpretation="Monitors downside protection. Advisory rebalancing aims to limit drawdowns during crisis periods compared to equal-weight baselines."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.crisis_drawdown}
              csvFilename={`${universe}_crisis_drawdowns.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.crisis_drawdown.map(d => d.strategy), scaleType: 'band' }]}
                  series={[{ data: backtest.data.crisis_drawdown.map(d => d.crisisDrawdown), color: '#ef4444', label: 'Crisis Drawdown %' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 83 */}
            <PlotCard
              title="83. CVaR Reduction by Sector Universe"
              description="Displays CVaR reduction percentages across major GICS sector universes."
              advisoryInterpretation="Measures risk reduction across sectors. Highlights sectors where rebalancing provides the most significant risk mitigation."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.sector_cvar_reduction}
              csvFilename={`${universe}_sector_cvar_reduction.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.sector_cvar_reduction.map(d => d.sector), scaleType: 'band' }]}
                  series={[{ data: backtest.data.sector_cvar_reduction.map(d => d.cvarReductionPercent), color: '#10b981', label: 'CVaR Reduction %' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 84 */}
            <PlotCard
              title="84. Rolling Sharpe Ratio Plot"
              description="Tracks the rolling Sharpe ratio of the advisory portfolio over the backtest window."
              advisoryInterpretation="Monitors consistency of risk-adjusted returns. Rising values indicate improving return generation relative to risk."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.rolling_sharpe}
              csvFilename={`${universe}_rolling_sharpe.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <LineChart
                  xAxis={[{ data: getDates(backtest.data.rolling_sharpe), scaleType: 'time' }]}
                  series={[{ data: getSeriesDataArray(backtest.data.rolling_sharpe, 'rollingSharpe'), label: 'Rolling Sharpe Ratio', color: '#10b981', showMark: false }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 85 */}
            <PlotCard
              title="85. Ablation Study Comparison"
              description="Ablation study evaluating the impact of removing graph centrality, regime adaptation, or HITL constraints."
              advisoryInterpretation="Identifies key model drivers. Demonstrates how each component contributes to overall risk-adjusted returns."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.ablation_study}
              csvFilename={`${universe}_ablation.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.ablation_study.map(d => d.model), scaleType: 'band' }]}
                  series={[
                    { data: backtest.data.ablation_study.map(d => d.sharpe), label: 'Sharpe Ratio', color: '#10b981' },
                    { data: backtest.data.ablation_study.map(d => d.maxDrawdown), label: 'Max Drawdown %', color: '#ef4444' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 86 */}
            <PlotCard
              title="86. Transaction Cost Impact Plot"
              description="Estimated cost drag from transaction costs and rebalancing slippage."
              advisoryInterpretation="Monitors rebalancing costs. Higher turnover strategies may incur higher cost drag, reducing net returns."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.cost_drag}
              csvFilename={`${universe}_cost_drag.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.cost_drag.map(d => d.strategy), scaleType: 'band' }]}
                  series={[{ data: backtest.data.cost_drag.map(d => d.costDrag), color: '#ef4444', label: 'Cost Drag %' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 87 */}
            <PlotCard
              title="87. Turnover Comparison Plot"
              description="Average portfolio turnover comparison across strategies."
              advisoryInterpretation="Measures portfolio stability. G-CVaR seeks to limit turnover to control rebalancing costs."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.turnover_comparison}
              csvFilename={`${universe}_turnover_comp.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.turnover_comparison.map(d => d.strategy), scaleType: 'band' }]}
                  series={[{ data: backtest.data.turnover_comparison.map(d => d.averageTurnover), color: '#3b82f6', label: 'Avg Turnover %' }]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />

            {/* Plot 88 */}
            <PlotCard
              title="88. IS vs OOS Sharpe Plot"
              description="Compares In-Sample (IS) and Out-of-Sample (OOS) Sharpe ratios across universes."
              advisoryInterpretation="Evaluates model stability. A narrow gap between IS and OOS ratios suggests the strategy is robust to overfitting."
              loading={backtest.loading}
              error={backtest.error}
              data={backtest.data?.is_oos_sharpe}
              csvFilename={`${universe}_oos_sharpe.csv`}
              isMock={backtest.data?.is_mock}
              renderChart={() => (
                <BarChart
                  xAxis={[{ data: backtest.data.is_oos_sharpe.map(d => d.universe), scaleType: 'band' }]}
                  series={[
                    { data: backtest.data.is_oos_sharpe.map(d => d.inSampleSharpe), label: 'In-Sample Sharpe', color: '#10b981' },
                    { data: backtest.data.is_oos_sharpe.map(d => d.outOfSampleSharpe), label: 'Out-of-Sample Sharpe', color: '#3b82f6' }
                  ]}
                  height={240}
                  margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
                />
              )}
            />
          </AnalyticsTabLayout>
        )}

      </Box>
    </Box>
  );
}
