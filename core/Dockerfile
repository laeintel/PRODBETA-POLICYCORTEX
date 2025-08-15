# Multi-stage build for Rust backend (single-crate layout)
FROM rust:alpine AS builder

# Build dependencies (include OpenSSL static libraries for Azure SDK)
RUN apk add --no-cache build-base pkgconfig openssl-dev openssl-libs-static

WORKDIR /app

# Copy manifest and source
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY migrations ./migrations

# Use SQLX offline to avoid DB requirement during image build
ENV SQLX_OFFLINE=true

# Build with cache mounts
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    (test -f target/release/policycortex-core && cp target/release/policycortex-core /app/policycortex-core || cp target/release/policycortex_core /app/policycortex-core)

# Runtime stage
FROM alpine:3.19

RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    wget \
    netcat-openbsd

# Create non-root user with home directory
RUN addgroup -g 1000 policycortex && \
    adduser -D -u 1000 -G policycortex -h /home/policycortex policycortex

# Copy binary from builder (normalized path)
COPY --from=builder /app/policycortex-core /usr/local/bin/policycortex-core

RUN chown policycortex:policycortex /usr/local/bin/policycortex-core && \
    mkdir -p /home/policycortex/.azure && \
    chown -R policycortex:policycortex /home/policycortex

USER policycortex

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

CMD ["policycortex-core"]