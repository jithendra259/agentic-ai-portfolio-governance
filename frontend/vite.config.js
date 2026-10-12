import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5000,
    host: true,
    allowedHosts: true,
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    }
  },
  build: {
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('@mui/x-charts')) {
              return 'vendor-mui-charts';
            }
            if (id.includes('@mui/material') || id.includes('@mui/icons-material') || id.includes('@emotion')) {
              return 'vendor-mui-core';
            }
            if (id.includes('katex') || id.includes('remark-math') || id.includes('rehype-katex')) {
              return 'vendor-katex';
            }
            if (id.includes('react-markdown') || id.includes('remark-gfm')) {
              return 'vendor-markdown';
            }
          }
        }
      }
    }
  }
})
