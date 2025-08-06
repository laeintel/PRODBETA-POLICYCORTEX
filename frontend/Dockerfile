# Node.js runtime for React application
FROM node:18-alpine

# Install security updates and required packages
RUN apk update && apk upgrade && apk add --no-cache curl

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev for build)
RUN npm install

# Copy source code
COPY . .

# Create production environment file if it doesn't exist
# Note: If .env.production exists in the build context, it will be copied with COPY . .
RUN touch .env.production

# Build arguments for environment variables
ARG NODE_ENV=production
ARG VITE_API_BASE_URL
ARG VITE_WS_URL
ARG VITE_AZURE_CLIENT_ID
ARG VITE_AZURE_TENANT_ID
ARG VITE_AZURE_REDIRECT_URI
ARG VITE_APP_VERSION

# Set environment variables
ENV NODE_ENV=$NODE_ENV
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_AZURE_CLIENT_ID=$VITE_AZURE_CLIENT_ID
ENV VITE_AZURE_TENANT_ID=$VITE_AZURE_TENANT_ID
ENV VITE_AZURE_REDIRECT_URI=$VITE_AZURE_REDIRECT_URI
ENV VITE_APP_VERSION=$VITE_APP_VERSION
ENV PORT=8080

# Build the application without TypeScript checking (for faster Docker builds)
RUN npm run build:no-tsc

# Install serve globally for production serving
RUN npm install -g serve

# Copy and setup entrypoint script before switching user
COPY scripts/generate-config.sh /app/
RUN chmod +x /app/generate-config.sh

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Change ownership (including the script)
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 8080

# Health check (serve static files, so check root)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/ || exit 1

# Serve the built application with Node.js
ENTRYPOINT ["/bin/sh", "-c", "/app/generate-config.sh && serve -s dist -l 8080 --no-clipboard"]