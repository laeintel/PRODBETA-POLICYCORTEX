# Multi-stage build for Rust backend with cargo-chef for optimized dependency caching
# This resolves compilation issues and significantly improves build performance

# Stage 1: Planner - prepare cargo dependencies
FROM lukemathwalker/cargo-chef:latest-rust-1.75-alpine AS chef
WORKDIR /app

# Install build dependencies for Azure SDK and other crates
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    openssl-libs-static \
    protobuf-dev \
    build-base

# Stage 2: Cache dependencies using cargo-chef
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
# Create dummy src to generate recipe
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Build dependencies (cached layer)
FROM chef AS cacher
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this layer is cached unless dependencies change
RUN cargo chef cook --release --recipe-path recipe.json

# Stage 4: Build the application
FROM chef AS builder

# Copy cached dependencies
COPY --from=cacher /app/target target
COPY --from=cacher $CARGO_HOME $CARGO_HOME

# Copy actual source code
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY migrations ./migrations

# Set environment variables for build
ENV SQLX_OFFLINE=true
ENV RUSTFLAGS="-C target-feature=-crt-static"

# Build the actual binary with proper error handling
RUN cargo build --release --bin policycortex-core && \
    # Verify the binary exists and is executable
    test -f target/release/policycortex-core && \
    cp target/release/policycortex-core /app/policycortex-core && \
    strip /app/policycortex-core && \
    chmod +x /app/policycortex-core

# Stage 5: Runtime image (minimal)
FROM alpine:3.19 AS runtime

# Install runtime dependencies only
RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    libssl3 \
    tzdata \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user with proper home directory
RUN addgroup -g 1000 policycortex && \
    adduser -D -u 1000 -G policycortex -h /home/policycortex -s /bin/sh policycortex

# Create necessary directories with proper permissions
RUN mkdir -p /home/policycortex/.azure /var/log/policycortex /var/cache/policycortex && \
    chown -R policycortex:policycortex /home/policycortex /var/log/policycortex /var/cache/policycortex

# Copy binary from builder
COPY --from=builder --chown=policycortex:policycortex /app/policycortex-core /usr/local/bin/policycortex-core

# Copy migrations if needed
COPY --from=builder --chown=policycortex:policycortex /app/migrations /home/policycortex/migrations

# Set working directory
WORKDIR /home/policycortex

# Switch to non-root user
USER policycortex

# Expose application port
EXPOSE 8080

# Health check with proper timeout and retry logic
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set proper entrypoint
ENTRYPOINT ["/usr/local/bin/policycortex-core"]