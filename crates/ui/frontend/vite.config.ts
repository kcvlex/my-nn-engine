import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import checker from 'vite-plugin-checker';

export default defineConfig({
  plugins: [
    vue(),
    checker({
      vueTsc: true,
      eslint: {
        useFlatConfig: true,
        lintCommand: 'eslint .',
      },
    }),
  ],
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
});
