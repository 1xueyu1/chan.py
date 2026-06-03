import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 后端 quant.server 默认在 3000 端口；dev 时 /api/* 代理过去，避免 CORS。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:3000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
