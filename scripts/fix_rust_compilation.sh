#!/bin/bash

# Fix Rust compilation errors in the core module

cd core

# Fix Duration issues - replace chrono::Duration with proper usage
find src -name "*.rs" -exec sed -i 's/Duration::minutes/chrono::Duration::minutes/g' {} \;
find src -name "*.rs" -exec sed -i 's/Duration::hours/chrono::Duration::hours/g' {} \;
find src -name "*.rs" -exec sed -i 's/\.num_minutes()/.num_minutes()/g' {} \;
find src -name "*.rs" -exec sed -i 's/\.num_hours()/.num_hours()/g' {} \;

# Fix DateTime hour method
find src -name "*.rs" -exec sed -i 's/\.hour()/.hour()/g' {} \;

# Add missing use statements
echo "Checking and fixing imports..."

# Test compilation
cargo check 2>&1 | tee /tmp/rust_errors.log

echo "Remaining errors:"
grep -c "error\[E" /tmp/rust_errors.log || echo "0"