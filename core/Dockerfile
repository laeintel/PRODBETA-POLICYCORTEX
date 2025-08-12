# Multi-stage build for Rust backend
FROM rust:alpine AS builder

# Install build dependencies (for openssl-sys vendored and general build tools)
# - build-base: gcc, make, libc-dev (musl)
# - perl: required by OpenSSL build scripts
# - pkgconfig: used by many -sys crates to discover libs
# - openssl-dev: fallback headers/libs when vendored is disabled
RUN apk add --no-cache build-base perl pkgconfig openssl-dev

# Create app directory
WORKDIR /app

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main to cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source code
COPY src ./src


# Build the actual application
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    netcat-openbsd \
    openssl

# Create non-root user with home directory
RUN addgroup -g 1000 policycortex && \
    adduser -D -u 1000 -G policycortex -h /home/policycortex policycortex

# Copy binary from builder
COPY --from=builder /app/target/release/policycortex-core /usr/local/bin/policycortex-core

# Set ownership
RUN chown policycortex:policycortex /usr/local/bin/policycortex-core

# Create .azure directory for credentials
RUN mkdir -p /home/policycortex/.azure && \
    chown -R policycortex:policycortex /home/policycortex

# Switch to non-root user
USER policycortex

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["policycortex-core"]