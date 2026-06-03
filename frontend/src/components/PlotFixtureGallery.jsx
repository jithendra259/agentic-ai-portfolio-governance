import { Box, Typography } from '@mui/material';
import InlineChart from './InlineChart';

const PLOT_FIXTURES = [
  { id: 'test-line', label: 'Line' },
  { id: 'test-pie', label: 'Pie' },
  { id: 'test-bar', label: 'Bar' },
  { id: 'test-scatter', label: 'Scatter' },
  { id: 'test-sankey', label: 'Sankey' },
  { id: 'test-sparkline', label: 'Sparkline' },
  { id: 'test-candlestick', label: 'Candlestick' },
];

export default function PlotFixtureGallery() {
  return (
    <Box className="plot-fixture-page">
      <Box className="plot-fixture-header">
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          Plot Fixture Gallery
        </Typography>
        <Typography variant="body2" sx={{ color: '#a9a9a9' }}>
          Deterministic backend fixtures for visual QA.
        </Typography>
      </Box>
      <Box className="plot-fixture-grid">
        {PLOT_FIXTURES.map((fixture) => (
          <Box className="plot-fixture-card" key={fixture.id} data-plot-fixture={fixture.id}>
            <Typography variant="overline" sx={{ color: '#a9a9a9', letterSpacing: 0 }}>
              {fixture.label}
            </Typography>
            <InlineChart plotId={fixture.id} />
          </Box>
        ))}
      </Box>
    </Box>
  );
}
