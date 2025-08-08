FROM rust:1.82 as builder

WORKDIR /app

# Copy manifests (Cargo.lock is optional)
COPY Cargo.toml ./
COPY Cargo.lock* ./

# Build dependencies (this is cached)
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy source code
COPY src ./src

# Build application
RUN touch src/main.rs
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/policycortex-core /app/policycortex-core

EXPOSE 8080

CMD ["./policycortex-core"]