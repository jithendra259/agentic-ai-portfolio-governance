import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
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

export default function InteractivePlot({ data, title }) {
  // `data` is expected to be an array of objects: [{ date: '2023-01-01', ticker: 'AAPL', value: 150 }, ...]
  
  // Pivot data for Recharts: [{ date: '2023-01-01', AAPL: 150, MSFT: 250 }, ...]
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return { lines: [], pData: [] };
    
    const datesMap = new Map();
    const keys = new Set();
    
    data.forEach(item => {
      keys.add(item.ticker);
      if (!datesMap.has(item.date)) {
        datesMap.set(item.date, { date: item.date });
      }
      datesMap.get(item.date)[item.ticker] = item.value;
    });
    
    const pData = Array.from(datesMap.values()).sort((a, b) => new Date(a.date) - new Date(b.date));
    return { lines: Array.from(keys), pData };
  }, [data]);

  if (chartData.pData.length === 0) {
    return null;
  }

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
        overflow: 'hidden'
      }}
    >
      <Typography variant="h6" align="center" gutterBottom color="text.primary">
        {title || 'Interactive Plot'}
      </Typography>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={chartData.pData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
          <XAxis 
            dataKey="date" 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickMargin={10}
            minTickGap={30}
          />
          <YAxis 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af', fontSize: 12 }} 
            domain={['auto', 'auto']}
            tickFormatter={(val) => {
              // Simple formatting: if values are small (like log returns), show 2 decimals
              if (Math.abs(val) < 2 && val !== 0) return val.toFixed(3);
              if (val > 1000) return (val/1000).toFixed(1) + 'k';
              return val.toFixed(1);
            }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6' }}
            itemStyle={{ color: '#e5e7eb' }}
            labelStyle={{ color: '#9ca3af', marginBottom: '5px' }}
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          {chartData.lines.map((ticker, i) => (
            <Line 
              key={ticker}
              type="monotone" 
              dataKey={ticker} 
              stroke={COLORS[i % COLORS.length]} 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, strokeWidth: 0 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
}
