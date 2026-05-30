import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import ChatInterface from './components/ChatInterface';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#0b1020', // Deep dark blue-black
      paper: '#111827',   // Slightly lighter gray-blue
    },
    primary: {
      main: '#3b82f6', // Bright blue
      dark: '#1d4ed8',
    },
    secondary: {
      main: '#10b981', // Emerald green
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
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
