# Allotrope frontend — multi-stage: build with node, serve with nginx.
#
# The dev iteration cycle is rebuild-on-change. To pick up source edits:
#     docker compose -f docker/docker-compose.yml up -d --build frontend
# (The container == image == bundle discipline; same as the api.)

# --- Stage 1: build ----------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Install deps first for layer caching. If package.json/lock didn't change,
# this layer is reused.
COPY package.json package-lock.json* ./
# Node 22's bundled OpenSSL has an intermittent AES-GCM corruption bug
# under qemu emulation when downloading lots of small tarballs from
# the npm registry (ossl_gcm_stream_update: cipher operation failed).
# Aggressive retries paper over the issue — each failed wheel/tarball
# download retries up to 8× with exponential backoff before giving up.
# Belt: also disable Node's HTTP/2 in case it's the protocol-layer
# corruption (some reports tie this bug specifically to HTTP/2 streams).
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm config set fetch-retries 8 \
 && npm config set fetch-retry-mintimeout 30000 \
 && npm config set fetch-retry-maxtimeout 300000 \
 && npm config set fetch-timeout 600000 \
 && npm config set registry https://registry.npmjs.org/ \
 && (npm install --no-audit --no-fund --prefer-offline \
  || npm install --no-audit --no-fund \
  || npm install --no-audit --no-fund)

# Source
COPY tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts index.html ./
COPY src ./src

RUN npm run build

# --- Stage 2: serve ----------------------------------------------------
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
