import React from 'react';
import { Card, CardContent, Typography, Box, Skeleton, Alert, Button, Accordion, AccordionSummary, AccordionDetails, Stack, Tooltip, IconButton } from '@mui/material';
import { ChevronDown, Download, HelpCircle } from 'lucide-react';

// ---------------------------------------------------------------------------
// CSV Downloader Helper
// ---------------------------------------------------------------------------
function downloadCSV(data, filename = 'plot_data.csv') {
  if (!data || !Array.isArray(data) || data.length === 0) return;
  
  const headers = Object.keys(data[0]);
  const csvRows = [headers.join(',')];
  
  data.forEach(row => {
    const values = headers.map(header => {
      const val = row[header];
      const escaped = ('' + (val !== null && val !== undefined ? val : '')).replace(/"/g, '\\"');
      return `"${escaped}"`;
    });
    csvRows.push(values.join(','));
  });
  
  const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ---------------------------------------------------------------------------
// LoadingPlotSkeleton
// ---------------------------------------------------------------------------
export function LoadingPlotSkeleton() {
  return (
    <Box sx={{ width: '100%', py: 2 }}>
      <Skeleton variant="text" width="60%" height={24} sx={{ mb: 1, bgcolor: '#262626' }} />
      <Skeleton variant="text" width="40%" height={16} sx={{ mb: 2, bgcolor: '#262626' }} />
      <Skeleton variant="rectangular" width="100%" height={240} sx={{ borderRadius: '8px', mb: 2, bgcolor: '#1a1a1a' }} />
      <Skeleton variant="text" width="95%" height={16} sx={{ mb: 0.5, bgcolor: '#262626' }} />
      <Skeleton variant="text" width="80%" height={16} sx={{ bgcolor: '#262626' }} />
    </Box>
  );
}

// ---------------------------------------------------------------------------
// PlotCard Main Component
// ---------------------------------------------------------------------------
export default function PlotCard({
  title,
  description,
  advisoryInterpretation,
  loading = false,
  error = '',
  data = null,
  renderChart,
  csvFilename = 'data.csv',
  isMock = false
}) {
  const [expanded, setExpanded] = React.useState(false);

  const handleDownload = () => {
    if (data) {
      downloadCSV(Array.isArray(data) ? data : [data], csvFilename);
    }
  };

  return (
    <Card
      elevation={2}
      sx={{
        bgcolor: '#141414',
        border: '1px solid #262626',
        borderRadius: '10px',
        overflow: 'hidden',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        transition: 'border-color 0.2s ease',
        '&:hover': {
          borderColor: '#404040'
        }
      }}
    >
      {/* Title Bar */}
      <Box
        sx={{
          px: 2,
          pt: 2,
          pb: 1.5,
          borderBottom: '1px solid #1f1f1f',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start'
        }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, color: 'text.primary', letterSpacing: '0.01em', lineHeight: 1.25, display: 'flex', alignItems: 'center', gap: 1 }}>
            {title}
            {isMock && (
              <Tooltip title="Real stock prices are currently missing in the database. Utilizing calibrated simulated data to render calculations.">
                <Typography variant="caption" sx={{ color: '#f59e0b', bgcolor: 'rgba(245, 158, 11, 0.08)', px: 0.8, py: 0.2, borderRadius: '4px', border: '1px solid rgba(245, 158, 11, 0.25)', fontWeight: 600 }}>
                  Sample data
                </Typography>
              </Tooltip>
            )}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, lineHeight: 1.2 }}>
            {description}
          </Typography>
        </Box>

        <Stack direction="row" spacing={0.5} sx={{ ml: 1, flexShrink: 0 }}>
          <Tooltip title="View academic formula details">
            <IconButton size="small" onClick={() => setExpanded(!expanded)} sx={{ color: 'text.secondary', p: 0.5 }}>
              <HelpCircle size={15} />
            </IconButton>
          </Tooltip>
          {data && (
            <Tooltip title="Download raw data as CSV">
              <IconButton size="small" onClick={handleDownload} sx={{ color: 'text.secondary', p: 0.5 }}>
                <Download size={15} />
              </IconButton>
            </Tooltip>
          )}
        </Stack>
      </Box>

      {/* Expandable Explanation Panel */}
      {expanded && (
        <Box sx={{ p: 2, bgcolor: '#1a1a1a', borderBottom: '1px solid #262626' }}>
          <Typography variant="caption" sx={{ color: 'text.primary', fontWeight: 600, display: 'block', mb: 0.5 }}>
            Methodological Explanation
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.4 }}>
            {description}
          </Typography>
        </Box>
      )}

      {/* Card Content & Chart Area */}
      <CardContent sx={{ p: 2.5, flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: '280px' }}>
        {loading ? (
          <LoadingPlotSkeleton />
        ) : error ? (
          <Alert severity="error" sx={{ bgcolor: 'rgba(239, 68, 68, 0.06)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
            {error}
          </Alert>
        ) : !data ? (
          <Alert severity="info" sx={{ bgcolor: 'rgba(59, 130, 246, 0.06)', color: '#9cd2f6', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
            Empty payload received. Ticker universe data may be loading.
          </Alert>
        ) : (
          <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            {/* The Plot Canvas */}
            <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              {renderChart()}
            </Box>

            {/* Advisory Interpretation Statement */}
            {advisoryInterpretation && (
              <Box sx={{ mt: 2, p: 1.25, bgcolor: '#1a1c22', borderLeft: '3px solid #60a5fa', borderRadius: '0 4px 4px 0' }}>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontStyle: 'italic', lineHeight: 1.35 }}>
                  <strong style={{ color: '#e5e7eb', fontStyle: 'normal', display: 'block', marginBottom: '2px', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Advisory Guidance</strong>
                  {advisoryInterpretation}
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
