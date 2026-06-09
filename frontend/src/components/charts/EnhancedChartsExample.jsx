import React, { useState, useCallback, useMemo } from 'react';
import { Box, Paper, Grid, Tab, Tabs, Typography } from '@mui/material';
import EnhancedBarChart from './bar/EnhancedBarChart';
import EnhancedLineChart from './line/EnhancedLineChart';
import EnhancedPieChart from './pie/EnhancedPieChart';
import { useResponsiveChartWidth } from './common/useResponsiveChartWidth';

/**
 * Comprehensive Example: Using All Enhanced Chart Components
 * Demonstrates best practices, responsive design, and all features
 */
export function EnhancedChartsExample() {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedItems, setSelectedItems] = useState({});
  const [chartRef, chartWidth] = useResponsiveChartWidth(800, 400);

  // Sample data for Bar Chart
  const barChartSpec = useMemo(() => ({
    title: 'Quarterly Sales Performance',
    xLabel: 'Quarter',
    yLabel: 'Revenue (USD)',
    layout: 'vertical',
    series: [
      {
        label: '2023',
        data: [12000, 19000, 23000, 29000],
        color: '#3b82f6'
      },
      {
        label: '2024',
        data: [13000, 21000, 25000, 31000],
        color: '#10b981'
      }
    ],
    xAxis: [
      {
        id: 'x-axis',
        data: ['Q1', 'Q2', 'Q3', 'Q4'],
        scaleType: 'band'
      }
    ],
    valueFormatter: {
      type: 'currency',
      locale: 'en-US',
      currency: 'USD',
      precision: 0
    },
    showGrid: true,
    showLegend: true
  }), []);

  // Sample data for Line Chart
  const lineChartSpec = useMemo(() => ({
    title: 'Monthly Revenue Trend',
    xLabel: 'Month',
    yLabel: 'Revenue (USD)',
    xAxisType: 'point',
    series: [
      {
        label: 'Actual',
        data: [
          { x: 'Jan', value: 15000 },
          { x: 'Feb', value: 18000 },
          { x: 'Mar', value: 21000 },
          { x: 'Apr', value: 19000 },
          { x: 'May', value: 24000 },
          { x: 'Jun', value: 27000 }
        ],
        color: '#3b82f6',
        curve: 'linear',
        area: false,
        showMark: true
      },
      {
        label: 'Target',
        data: [
          { x: 'Jan', value: 16000 },
          { x: 'Feb', value: 16500 },
          { x: 'Mar', value: 17000 },
          { x: 'Apr', value: 17500 },
          { x: 'May', value: 18000 },
          { x: 'Jun', value: 18500 }
        ],
        color: '#ef4444',
        curve: 'linear',
        area: false,
        showMark: true
      }
    ],
    valueFormatter: {
      type: 'currency',
      locale: 'en-US',
      currency: 'USD',
      precision: 0
    },
    showGrid: true,
    showLegend: true
  }), []);

  // Sample data for Pie Chart
  const pieChartSpec = useMemo(() => ({
    title: 'Revenue by Product Category',
    series: [
      {
        data: [
          { id: 'software', label: 'Software', value: 450000 },
          { id: 'services', label: 'Services', value: 380000 },
          { id: 'hardware', label: 'Hardware', value: 290000 },
          { id: 'support', label: 'Support', value: 180000 },
          { id: 'consulting', label: 'Consulting', value: 150000 }
        ],
        innerRadius: 50,
        outerRadius: 120,
        paddingAngle: 2,
        cornerRadius: 4,
        highlightScope: {
          faded: 'global',
          highlighted: 'item'
        }
      }
    ],
    colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
    valueFormatType: 'currency',
    valueFormatter: {
      type: 'currency',
      locale: 'en-US',
      currency: 'USD',
      precision: 0
    }
  }), []);

  // Handle chart interactions
  const handleBarChartClick = useCallback((event, params) => {
    setSelectedItems(prev => ({
      ...prev,
      bar: params
    }));
    console.log('Bar chart clicked:', params);
  }, []);

  const handleLineChartClick = useCallback((event, params) => {
    setSelectedItems(prev => ({
      ...prev,
      line: params
    }));
    console.log('Line chart clicked:', params);
  }, []);

  const handlePieChartClick = useCallback((event, params) => {
    setSelectedItems(prev => ({
      ...prev,
      pie: params
    }));
    console.log('Pie chart clicked:', params);
  }, []);

  return (
    <Box sx={{ width: '100%', p: 2 }} ref={chartRef}>
      <Paper elevation={0} sx={{ p: 2 }}>
        <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
          Enhanced Charts System - Complete Example
        </Typography>

        {/* Tab Navigation */}
        <Tabs 
          value={activeTab} 
          onChange={(e, v) => setActiveTab(v)}
          sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Bar Chart" />
          <Tab label="Line Chart" />
          <Tab label="Pie Chart" />
          <Tab label="Comparison" />
        </Tabs>

        {/* Tab Content */}
        {activeTab === 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
              Vertical Bar Chart with Multiple Series
            </Typography>
            <EnhancedBarChart
              spec={barChartSpec}
              onItemClick={handleBarChartClick}
            />
            {selectedItems.bar && (
              <Typography 
                variant="caption" 
                sx={{ mt: 1, display: 'block', color: 'text.secondary' }}
              >
                Selected: {JSON.stringify(selectedItems.bar)}
              </Typography>
            )}
          </Box>
        )}

        {activeTab === 1 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
              Line Chart with Target Comparison
            </Typography>
            <EnhancedLineChart
              spec={lineChartSpec}
              onItemClick={handleLineChartClick}
            />
            {selectedItems.line && (
              <Typography 
                variant="caption" 
                sx={{ mt: 1, display: 'block', color: 'text.secondary' }}
              >
                Selected: {JSON.stringify(selectedItems.line)}
              </Typography>
            )}
          </Box>
        )}

        {activeTab === 2 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
              Pie Chart - Revenue Distribution
            </Typography>
            <EnhancedPieChart
              spec={pieChartSpec}
              onItemClick={handlePieChartClick}
            />
            {selectedItems.pie && (
              <Typography 
                variant="caption" 
                sx={{ mt: 1, display: 'block', color: 'text.secondary' }}
              >
                Selected: {JSON.stringify(selectedItems.pie)}
              </Typography>
            )}
          </Box>
        )}

        {activeTab === 3 && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
                  Sales Performance
                </Typography>
                <EnhancedBarChart spec={barChartSpec} />
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
                  Category Distribution
                </Typography>
                <EnhancedPieChart spec={pieChartSpec} />
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
                  Revenue Trend
                </Typography>
                <EnhancedLineChart spec={lineChartSpec} />
              </Box>
            </Grid>
          </Grid>
        )}

        {/* Information Panel */}
        <Paper 
          elevation={0} 
          sx={{ 
            mt: 3, 
            p: 2, 
            bgcolor: 'action.hover',
            borderRadius: 1
          }}
        >
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Features Demonstrated:
          </Typography>
          <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
            <li>Responsive design with automatic resizing</li>
            <li>Value formatting (currency with thousands separator)</li>
            <li>Multiple series and axis support</li>
            <li>Theme-aware styling (dark/light mode compatible)</li>
            <li>Interactive selection with callbacks</li>
            <li>Grid and legend control</li>
            <li>Accessibility with proper ARIA labels</li>
            <li>Error and loading state support</li>
          </ul>
        </Paper>
      </Paper>
    </Box>
  );
}

export default EnhancedChartsExample;
