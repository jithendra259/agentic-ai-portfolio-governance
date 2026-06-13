import { Box, Tab, Tabs } from '@mui/material';

import { DASHBOARD_TABS } from './analyticsDashboardModel';

export default function AnalyticsDashboardTabs({ activeTab, onTabChange }) {
  return (
    <Box sx={{ borderBottom: '1px solid #262626', bgcolor: '#121212', flexShrink: 0 }}>
      <Tabs
        value={activeTab}
        onChange={onTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          minHeight: '48px',
          '& .MuiTab-root': {
            color: '#B4B4B4',
            textTransform: 'none',
            fontWeight: 600,
            fontSize: '0.85rem',
            minHeight: '48px',
            px: 2.5,
            '&.Mui-selected': {
              color: '#f59e0b',
            },
          },
          '& .MuiTabs-indicator': {
            bgcolor: '#f59e0b',
          },
        }}
      >
        {DASHBOARD_TABS.map((label) => (
          <Tab key={label} label={label} />
        ))}
      </Tabs>
    </Box>
  );
}
