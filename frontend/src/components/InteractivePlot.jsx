import React, { useMemo } from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { Box, Typography, Paper } from '@mui/material';

const COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
];

// Helper to format values cleanly on hover and axes
const formatValue = (val) => {
  if (val === null || val === undefined) return '';
  if (Math.abs(val) < 2 && val !== 0) return val.toFixed(3);
  if (val > 1000) return (val / 1000).toFixed(1) + 'k';
  if (val < -1000) return (val / 1000).toFixed(1) + 'k';
  return val.toFixed(1);
};

export default function InteractivePlot({ data, title }) {
  // Pivot data for MUI Charts: [{ date: new Date('2023-01-01'), AAPL: 150, MSFT: 250 }, ...]
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return { lines: [], pData: [] };
    
    const datesMap = new Map();
    const keys = new Set();
    
    data.forEach(item => {
      keys.add(item.ticker);
      if (!datesMap.has(item.date)) {
        datesMap.set(item.date, { date: new Date(item.date) });
      }
      datesMap.get(item.date)[item.ticker] = item.value;
    });
    
    const pData = Array.from(datesMap.values()).sort((a, b) => a.date - b.date);
    return { lines: Array.from(keys), pData };
  }, [data]);

  if (chartData.pData.length === 0) {
    return null;
  }

  // Create the series configuration array for MUI Charts
  const seriesConfig = chartData.lines.map((ticker, index) => ({
    dataKey: ticker,
    label: ticker,
    color: COLORS[index % COLORS.length],
    showMark: false,
    connectNulls: true, // Interpolate over missing data gracefully
    curve: 'monotoneX', // Smooth financial curves
    valueFormatter: formatValue,
    highlightScope: { highlight: 'series', fade: 'global' },
  }));

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        mt: 1, 
        mb: 2, 
        width: '100%', 
        height: 400, 
        bgcolor: 'background.paper',
        borderRadius: 2,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Typography variant="h6" align="center" gutterBottom color="text.primary">
        {title || 'Interactive Plot'}
      </Typography>
      
      <Box sx={{ flex: 1, minHeight: 0 }}>
        <LineChart
          dataset={chartData.pData}
          xAxis={[{ 
            dataKey: 'date', 
            scaleType: 'time',
            tickLabelStyle: { fill: '#9ca3af', fontSize: 12 },
            valueFormatter: (date) => {
              if (!date) return '';
              return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
            }
          }]}
          yAxis={[{
            tickLabelStyle: { fill: '#9ca3af', fontSize: 12 },
            valueFormatter: formatValue,
            domainLimit: 'nice', // Round axes instead of cutting off strictly at max value
          }]}
          series={seriesConfig}
          margin={{ top: 20, right: 30, left: 50, bottom: 30 }}
          grid={{ horizontal: true }}
          experimentalFeatures={{ enablePositionBasedPointerInteraction: true }}
          slotProps={{
            legend: {
              labelStyle: { fill: '#e5e7eb' },
            }
          }}
          sx={{
            '& .MuiChartsGrid-line': { stroke: '#374151', strokeDasharray: '4 4' },
          }}
        />
      </Box>
    </Paper>
  );
}
