import { createTheme } from '@mui/material/styles';

export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00C853',
    },
    secondary: {
      main: '#004D40',
    },
    background: {
      paper: '#121212',
      default: '#121212',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0bec5',
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#000000',
        },
      },
    },
  },
});

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#263238', // Dark navy blue
    },
    secondary: {
      main: '#37474F', // Smoky gray
    },
    background: {
      paper: '#ECEFF1',
      default: '#CFD8DC',
    },
    text: {
      primary: '#263238',
      secondary: '#455A64',
    },
  },
}); 