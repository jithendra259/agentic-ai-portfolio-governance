import React, { useState } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import ChatInterface from './components/ChatInterface';
import AnalyticsDashboard from './components/AnalyticsDashboard';


const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#0D0D0D',    // ChatGPT-style pure black
      paper: '#1A1A1A',      // Very dark gray for surfaces
    },
    text: {
      primary: '#ECECEC',    // Light text
      secondary: '#B4B4B4',  // Medium gray text
    },
    divider: '#404040',      // Dark gray borders
    action: {
      hover: 'rgba(255, 255, 255, 0.05)',
      disabled: '#666666',
    },
    primary: {
      main: '#FFFFFF',       // White for primary
      dark: '#B4B4B4',       // Gray for dark
    },
    secondary: {
      main: '#B4B4B4',       // Gray accents
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    color: '#ECECEC',
  },
  components: {
    MuiChatBox: {
      styleOverrides: {
        root: ({ theme }) => ({
          backgroundColor: '#0D0D0D',
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: 12,
          color: '#ECECEC',
        }),
        conversationsPane: ({ theme }) => ({
          backgroundColor: '#0D0D0D',
          borderRight: `1px solid ${theme.palette.divider}`,
        }),
        messagesPane: {
          backgroundColor: '#0D0D0D',
        },
      },
    },
    MuiChatMessage: {
      styleOverrides: {
        bubble: ({ theme }) => ({
          borderRadius: 12,
          padding: '12px 16px',
          color: '#ECECEC',
          backgroundColor: '#2A2A2A',
          '&:hover': {
            backgroundColor: '#353535',
          },
        }),
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          color: '#ECECEC',
          borderColor: '#404040',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            borderColor: '#666666',
          },
        },
      },
    },
    MuiInputBase: {
      styleOverrides: {
        root: {
          color: '#ECECEC',
          backgroundColor: '#1A1A1A',
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: '#404040',
          },
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#666666',
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#FFFFFF',
          },
        },
      },
    },
  },
});

function App() {
  const [view, setView] = useState('chat');

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      {view === 'chat' ? (
        <ChatInterface setView={setView} />
      ) : (
        <AnalyticsDashboard setView={setView} />
      )}
    </ThemeProvider>
  );
}

export default App;

