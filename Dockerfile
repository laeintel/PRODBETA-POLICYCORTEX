# Multi-stage build for Rust workspace
FROM rust:alpine AS builder

# Install build dependencies
RUN apk add --no-cache build-base openssl-dev openssl-libs-static pkgconfig perl

# Set OpenSSL environment variables
ENV OPENSSL_DIR=/usr
ENV OPENSSL_LIB_DIR=/usr/lib
ENV OPENSSL_INCLUDE_DIR=/usr/include
ENV OPENSSL_STATIC=1
ENV PKG_CONFIG_ALLOW_CROSS=1

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build the workspace
RUN cargo build --release -p policycortex-api

# Runtime stage
FROM alpine:latest

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/policycortex /app/policycortex

# Copy any static files if needed
COPY --from=builder /app/crates/api/static ./static 2>/dev/null || true

EXPOSE 8080

CMD ["./policycortex"]