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
ENV PORT=80

# Build the application
RUN npm run build

# Install serve globally for production serving
RUN npm install -g serve

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Change ownership
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

# Serve the built application with Node.js
CMD ["serve", "-s", "dist", "-l", "80", "--no-clipboard"]