#!/bin/bash
NODE=/nix/store/bl6iwirn83qj9r8wng43kfdqd5mfahj8-nodejs-22.22.0/bin/node
VITE=/home/runner/workspace/frontend/node_modules/.pnpm/vite@8.0.14_yaml@2.5.1/node_modules/vite/bin/vite.js
export NODE_PATH="/home/runner/workspace/frontend/node_modules/.pnpm/vite@8.0.14_yaml@2.5.1/node_modules/vite/bin/node_modules:/home/runner/workspace/frontend/node_modules/.pnpm/vite@8.0.14_yaml@2.5.1/node_modules/vite/node_modules:/home/runner/workspace/frontend/node_modules/.pnpm/vite@8.0.14_yaml@2.5.1/node_modules:/home/runner/workspace/frontend/node_modules/.pnpm/node_modules"
cd /home/runner/workspace/frontend
exec "$NODE" "$VITE"
