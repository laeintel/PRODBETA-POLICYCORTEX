FROM node:20-alpine

WORKDIR /app

# Install Wrangler CLI for edge development
RUN npm install -g wrangler@latest

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy edge functions
COPY . .

EXPOSE 8787

# Start Wrangler dev server
CMD ["wrangler", "dev", "--local", "--port", "8787"]