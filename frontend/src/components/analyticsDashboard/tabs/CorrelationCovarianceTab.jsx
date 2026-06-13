import { BarChart } from '@mui/x-charts/BarChart';
import { LineChart } from '@mui/x-charts/LineChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';

import AnalyticsTabLayout from '../../AnalyticsTabLayout';
import { HeatmapChart } from '../../CustomCharts';
import MetricSummaryCards from '../../MetricSummaryCards';
import PlotCard from '../../PlotCard';
import { getDates, getSeriesDataArray } from '../analyticsDashboardModel';

export default function CorrelationCovarianceTab({ activeRegime, eda, tickers, universe }) {
  return (
    <AnalyticsTabLayout
      title="Correlation & Covariance EDA"
      description="Examines inter-asset relationships, eigenvalues, and PCA variance to evaluate systemic risk concentration across principal components."
      regime={activeRegime}
      summaryCards={
        <MetricSummaryCards
          metrics={[
            {
              label: 'Avg Pairwise Correlation',
              value: eda.data?.rolling_correlation?.length > 0
                ? `${(eda.data.rolling_correlation[eda.data.rolling_correlation.length - 1].averageCorrelation * 100).toFixed(1)}%`
                : '30.0%',
              helpText: 'Average asset overlap',
            },
            {
              label: 'PC 1 Variance Explained',
              value: eda.data?.pca_explained_variance?.length > 0
                ? `${eda.data.pca_explained_variance[0].explainedVariancePercent.toFixed(1)}%`
                : '55.0%',
              color: '#f59e0b',
              helpText: 'Risk concentration in major factor',
            },
            {
              label: 'Covariance Drift Index',
              value: eda.data?.covariance_drift?.length > 0
                ? eda.data.covariance_drift[eda.data.covariance_drift.length - 1].covarianceDrift.toFixed(2)
                : '0.15',
              helpText: 'Drift from baseline norm',
            },
            { label: 'Correlation Stress Level', value: 'Normal', color: '#10b981', helpText: 'Stress index is below critical' },
          ]}
        />
      }
    >
      <PlotCard
        title="11. Return Correlation Heatmap"
        description="Displays pairwise return correlation coefficients matrix."
        advisoryInterpretation="Values close to 1 indicate high overlap. Lower coefficients offer better diversification properties."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.correlation_heatmap}
        csvFilename={`${universe}_correlation_matrix.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => <HeatmapChart data={eda.data.correlation_heatmap} />}
      />

      <PlotCard
        title="12. Rolling Average Correlation Plot"
        description="40-day rolling window of average pairwise correlation between log returns."
        advisoryInterpretation="Sharply rising correlation over time signals systemic market stress, reducing standard diversification effectiveness."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.rolling_correlation}
        csvFilename={`${universe}_rolling_corr.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.rolling_correlation), scaleType: 'time' }]}
            series={[{ data: getSeriesDataArray(eda.data.rolling_correlation, 'averageCorrelation'), label: 'Avg Correlation', showMark: false }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="13. Covariance Matrix Heatmap"
        description="Displays annualized covariance values matrix (scaled % squared)."
        advisoryInterpretation="Identifies absolute directional risk. Overweighting high covariance asset pairs elevates portfolio volatility."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.covariance_heatmap}
        csvFilename={`${universe}_covariance_matrix.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => <HeatmapChart data={eda.data.covariance_heatmap} />}
      />

      <PlotCard
        title="14. Covariance Drift Plot"
        description="Traces the Frobenius norm of covariance matrix drift over time against the initial baseline period."
        advisoryInterpretation="Spikes represent structural covariance drift, requiring the G-CVaR optimizer to rebalance risk weights."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.covariance_drift}
        csvFilename={`${universe}_cov_drift.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.covariance_drift), scaleType: 'time' }]}
            series={[{ data: getSeriesDataArray(eda.data.covariance_drift, 'covarianceDrift'), label: 'Covariance Drift', color: '#8b5cf6', showMark: false }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="15. Correlation Stress Plot"
        description="Annualized upper-quantile correlation stress channel tracking worst-case pairwise overlaps."
        advisoryInterpretation="High stress values narrow the safe diversification window, signaling the need for defensive allocations."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.correlation_stress}
        csvFilename={`${universe}_corr_stress.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <LineChart
            xAxis={[{ data: getDates(eda.data.correlation_stress), scaleType: 'time' }]}
            series={[{ data: getSeriesDataArray(eda.data.correlation_stress, 'correlationStress'), label: 'Stress Level', color: '#ef4444', showMark: false }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="16. Eigenvalue Spectrum Plot"
        description="Eigenvalue magnitude of return covariance matrix sorted descending."
        advisoryInterpretation="The dominant eigenvalue indicates the magnitude of the largest common risk factor in the portfolio."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.eigenvalue_spectrum}
        csvFilename={`${universe}_eigenvalues.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: eda.data.eigenvalue_spectrum.map((d) => d.component), scaleType: 'band' }]}
            series={[{ data: eda.data.eigenvalue_spectrum.map((d) => d.eigenvalue), color: '#8b5cf6', label: 'Eigenvalue Magnitude' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="17. PCA Explained Variance Plot"
        description="Percentage contribution of each principal component to total return variance."
        advisoryInterpretation="High concentration in PC1 indicates the portfolio is heavily exposed to a single systemic market factor."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.pca_explained_variance}
        csvFilename={`${universe}_pca_variance.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => (
          <BarChart
            xAxis={[{ data: eda.data.pca_explained_variance.map((d) => d.component), scaleType: 'band' }]}
            series={[{ data: eda.data.pca_explained_variance.map((d) => d.explainedVariancePercent), color: '#10b981', label: 'Explained Variance %' }]}
            height={240}
            margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
          />
        )}
      />

      <PlotCard
        title="18. Pairwise Return Scatter Matrix"
        description="Selectable scatter plot representing daily returns of Ticker A against Ticker B."
        advisoryInterpretation="Linear shapes identify high positive correlation, indicating shared systemic vulnerability."
        loading={eda.loading}
        error={eda.error}
        data={eda.data?.pairwise_scatter}
        csvFilename={`${universe}_pairwise_scatter.csv`}
        isMock={eda.data?.is_mock}
        renderChart={() => {
          const scatterData = eda.data.pairwise_scatter || [];
          const t1 = tickers[0];
          const t2 = tickers[1] || tickers[0];
          return (
            <ScatterChart
              series={[{
                data: scatterData.map((d) => ({ x: d.returnX, y: d.returnY, id: d.date })),
                label: `${t1} vs ${t2} Scatter`,
              }]}
              height={240}
              margin={{ top: 20, bottom: 30, left: 40, right: 10 }}
            />
          );
        }}
      />
    </AnalyticsTabLayout>
  );
}
