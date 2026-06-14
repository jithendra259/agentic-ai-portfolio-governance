import React from 'react';
import { Box, Typography } from '@mui/material';
import { RadarChart } from '@mui/x-charts-premium/RadarChart';
import { getResponsiveChartHeight } from './radarChartUtils';
import { getMetricConfig } from '../../../utils/radarMetricsConfig.js';
import { formatRadarRawValue, normalizeRadarSeries } from '../../../utils/radarNormalizer.js';

export default function RadarChartRenderer({ spec }) {
  const height = getResponsiveChartHeight(spec, 360);
  const metrics = spec.radar?.metrics || spec.metrics || [];
  const shouldNormalize = spec.normalize !== false && spec.radar?.normalize !== false;
  const series = shouldNormalize
    ? normalizeRadarSeries(spec.series || [], metrics).map((entry) => ({
        ...entry,
        valueFormatter: (value, context) => {
          const index = context?.dataIndex ?? context?.itemData?.dataIndex;
          const metric = metrics[index];
          const rawValue = entry.rawData?.[index];
          return rawValue == null ? `${Number(value ?? 0).toFixed(1)}/100` : formatRadarRawValue(rawValue, metric);
        },
      }))
    : (spec.series || []);

  if (!metrics.length || !series.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>No radar data available.</Typography>
      </Box>
    );
  }

  // Build radar config — supports both string[] and MetricConfig[] ({ name, min, max })
  const radarConfig = {
    metrics: metrics.map((metric) => {
      const config = getMetricConfig(metric);
      return typeof metric === 'string'
        ? { name: metric, label: config.label }
        : { ...metric, label: metric.label || config.label };
    }),
    ...(spec.radar?.startAngle != null && { startAngle: spec.radar.startAngle }),
    ...(spec.radar?.labelGap != null && { labelGap: spec.radar.labelGap }),
    ...(spec.radar?.labelFormatter && { labelFormatter: spec.radar.labelFormatter }),
    max: shouldNormalize ? 100 : (spec.radar?.max ?? spec.max),
  };

  // Grid customization
  const gridShape = spec.shape || 'sharp'; // 'sharp' | 'circular'
  const gridDivisions = spec.divisions ?? 5;
  const stripeColor = spec.stripeColor ?? ((index) => (index % 2 === 1 ? 'rgba(148, 163, 184, 0.06)' : 'none'));

  // Highlight mode: 'axis' (default), 'series', or 'none'
  const highlightMode = spec.highlight || 'axis';

  // Color palette
  const colors = spec.colors || undefined;

  // Interaction handlers
  const handleAreaClick = spec.onAreaClick
    ? (event, identifier) => {
        const callback = typeof spec.onAreaClick === 'function' ? spec.onAreaClick : null;
        callback?.(event, identifier);
      }
    : undefined;

  const handleMarkClick = spec.onMarkClick
    ? (event, identifier) => {
        const callback = typeof spec.onMarkClick === 'function' ? spec.onMarkClick : null;
        callback?.(event, identifier);
      }
    : undefined;

  const handleAxisClick = spec.onAxisClick
    ? (event, identifier) => {
        const callback = typeof spec.onAxisClick === 'function' ? spec.onAxisClick : null;
        callback?.(event, identifier);
      }
    : undefined;

  return (
    <Box sx={{ width: '100%', minWidth: 280 }}>
      <RadarChart
        height={height}
        radar={radarConfig}
        series={series}
        hideLegend={spec.hideLegend ?? false}
        margin={spec.margin || { top: 24, right: 28, bottom: 30, left: 28 }}
        shape={gridShape}
        divisions={gridDivisions}
        stripeColor={stripeColor}
        highlight={highlightMode}
        colors={colors}
        skipAnimation={spec.skipAnimation ?? false}
        loading={spec.loading ?? false}
        showToolbar={spec.showToolbar ?? false}
        onAreaClick={handleAreaClick}
        onMarkClick={handleMarkClick}
        onAxisClick={handleAxisClick}
        disableKeyboardNavigation={spec.disableKeyboardNavigation ?? false}
        sx={spec.sx || undefined}
      />
    </Box>
  );
}
