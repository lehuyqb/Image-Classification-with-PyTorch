import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import ImageUpload from './components/ImageUpload';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ImageUpload />
    </ThemeProvider>
  );
};

export default App;
