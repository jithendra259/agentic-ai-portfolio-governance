import {
  Box,
  Button,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';
import { ArrowLeft, BarChart2, RefreshCw } from 'lucide-react';

const SELECT_SX = {
  color: '#ffffff',
  fontSize: '0.8rem',
  '.MuiOutlinedInput-notchedOutline': { borderColor: '#262626' },
};

const LABEL_SX = { color: '#B4B4B4', fontSize: '0.8rem' };

const SELECT_GROUPS = [
  {
    id: 'universe',
    label: 'Stock Universe',
    minWidth: 120,
    options: [
      ['U1', 'U1: Tech Mix (AAPL, MSFT...)'],
      ['U2', 'U2: Financials (BAC, GS...)'],
      ['U3', 'U3: Healthcare (JNJ, PFE...)'],
      ['U4', 'U4: Consumer (TSLA, MCD...)'],
      ['U5', 'U5: Industrials (GE, HON...)'],
    ],
  },
  {
    id: 'dates',
    label: 'Date Range',
    minWidth: 110,
    options: [
      ['2024', 'Year 2024 (Stress)'],
      ['2023', 'Year 2023 (Calm)'],
    ],
  },
  {
    id: 'strategy',
    label: 'Strategy',
    minWidth: 110,
    options: [
      ['G-CVaR', 'G-CVaR Model'],
      ['Standard-CVaR', 'Standard CVaR'],
      ['EqualWeight', 'Equal Weight'],
    ],
  },
];

export default function AnalyticsDashboardHeader({
  datePreset,
  onDatePresetChange,
  onRefresh,
  onStrategyChange,
  onUniverseChange,
  setView,
  strategy,
  universe,
}) {
  const values = { universe, dates: datePreset, strategy };
  const handlers = {
    universe: onUniverseChange,
    dates: onDatePresetChange,
    strategy: onStrategyChange,
  };

  return (
    <Box sx={{ minHeight: '64px', borderBottom: '1px solid #262626', display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 3, bgcolor: '#121212', flexShrink: 0 }}>
      <Stack direction="row" spacing={2} alignItems="center">
        <Button
          startIcon={<ArrowLeft size={16} />}
          onClick={() => setView('chat')}
          sx={{
            color: '#B4B4B4',
            borderColor: '#262626',
            textTransform: 'none',
            fontWeight: 600,
            fontSize: '0.85rem',
            '&:hover': {
              color: '#ffffff',
              bgcolor: 'rgba(255,255,255,0.05)',
            },
          }}
          variant="outlined"
          size="small"
        >
          Advisory Chat
        </Button>
        <Divider orientation="vertical" flexItem sx={{ bgcolor: '#262626' }} />
        <Typography variant="subtitle1" sx={{ fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 1 }}>
          <BarChart2 size={18} color="#f59e0b" />
          Advisory Portfolio Governance & Diversification System
        </Typography>
      </Stack>

      <Stack direction="row" spacing={2} alignItems="center">
        {SELECT_GROUPS.map((group) => (
          <FormControl key={group.id} size="small" variant="outlined" sx={{ minWidth: group.minWidth }}>
            <InputLabel id={`${group.id}-label`} sx={LABEL_SX}>{group.label}</InputLabel>
            <Select
              labelId={`${group.id}-label`}
              value={values[group.id]}
              onChange={(event) => handlers[group.id](event.target.value)}
              label={group.label}
              sx={SELECT_SX}
            >
              {group.options.map(([value, label]) => (
                <MenuItem key={value} value={value}>{label}</MenuItem>
              ))}
            </Select>
          </FormControl>
        ))}

        <Button
          size="small"
          onClick={onRefresh}
          sx={{ color: '#B4B4B4', minWidth: '40px', p: 1, border: '1px solid #262626', '&:hover': { bgcolor: 'rgba(255,255,255,0.05)' } }}
        >
          <RefreshCw size={16} />
        </Button>
      </Stack>
    </Box>
  );
}
