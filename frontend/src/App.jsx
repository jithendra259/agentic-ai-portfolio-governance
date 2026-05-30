import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import ChatInterface from './components/ChatInterface';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#121212', // Very dark background like the image
      paper: '#1a1a1e',   // Slightly lighter for surfaces
    },
    divider: '#2d2d35', // Subtle borders
    primary: {
      main: '#6366f1', // Indigo/purple accent
      dark: '#4f46e5',
    },
    secondary: {
      main: '#10b981', // Emerald green
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiChatBox: {
      styleOverrides: {
        root: ({ theme }) => ({
          border: '1px solid',
          borderColor: theme.palette.divider,
          backgroundColor: theme.palette.background.default,
          borderRadius: 8,
        }),
        conversationsPane: ({ theme }) => ({
          backgroundColor: '#18181b', // Darker pane
          borderRight: `1px solid ${theme.palette.divider}`,
        }),
      },
    },
    MuiChatMessage: {
      styleOverrides: {
        bubble: ({ theme }) => ({
          borderRadius: 16,
          padding: '12px 16px',
          color: '#ffffff',
        }),
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <ChatInterface />
    </ThemeProvider>
  );
}

export default App;
