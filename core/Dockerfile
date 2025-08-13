# Multi-stage build for Rust backend
FROM rust:alpine AS builder

# Install build dependencies
# - build-base: gcc, make, libc-dev (musl)
# - openssl-dev and openssl-libs-static: required for Azure SDK dependencies
# - pkgconfig: used by many -sys crates to discover libs
# - perl: required for building OpenSSL from source
RUN apk add --no-cache build-base openssl-dev openssl-libs-static pkgconfig perl

# Set OpenSSL environment variables to use system OpenSSL
ENV OPENSSL_DIR=/usr
ENV OPENSSL_LIB_DIR=/usr/lib
ENV OPENSSL_INCLUDE_DIR=/usr/include
ENV OPENSSL_STATIC=1
ENV PKG_CONFIG_ALLOW_CROSS=1

# Create app directory
WORKDIR /app

# Build workspace API crate instead of monolith core
# Copy workspace manifest and crates
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Use SQLX offline to avoid DB requirement during image build
ENV SQLX_OFFLINE=true

# Cache cargo registry and target for faster builds
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build -p policycortex-api --release

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    netcat-openbsd

# Create non-root user with home directory
RUN addgroup -g 1000 policycortex && \
    adduser -D -u 1000 -G policycortex -h /home/policycortex policycortex

# Copy binary from builder (api crate built as runtime binary)
COPY --from=builder /app/target/release/policycortex-api /usr/local/bin/policycortex-core

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