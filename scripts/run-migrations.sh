#!/bin/bash

# PolicyCortex Database Migration Runner
# Executes all SQL migrations in order

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-policycortex}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-./migrations}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "PolicyCortex Database Migration Runner"
echo "========================================"
echo ""

# Check if psql is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: psql is not installed${NC}"
    echo "Please install PostgreSQL client tools"
    exit 1
fi

# Check if migrations directory exists
if [ ! -d "$MIGRATIONS_DIR" ]; then
    echo -e "${RED}Error: Migrations directory not found: $MIGRATIONS_DIR${NC}"
    exit 1
fi

# Set PGPASSWORD environment variable for non-interactive authentication
export PGPASSWORD="$DB_PASSWORD"

# Test database connection
echo "Testing database connection..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Cannot connect to database${NC}"
    echo "Please check your database credentials and ensure PostgreSQL is running"
    exit 1
fi
echo -e "${GREEN}✓ Database connection successful${NC}"
echo ""

# Create migrations table if it doesn't exist
echo "Creating migrations tracking table..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64),
    execution_time_ms INTEGER
);
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create migrations table${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Migrations table ready${NC}"
echo ""

# Get list of migration files
MIGRATIONS=$(ls -1 "$MIGRATIONS_DIR"/*.sql 2>/dev/null | sort)

if [ -z "$MIGRATIONS" ]; then
    echo -e "${YELLOW}No migration files found in $MIGRATIONS_DIR${NC}"
    exit 0
fi

# Count migrations
TOTAL_MIGRATIONS=$(echo "$MIGRATIONS" | wc -l)
echo "Found $TOTAL_MIGRATIONS migration file(s)"
echo ""

# Execute migrations
EXECUTED=0
SKIPPED=0
FAILED=0

for MIGRATION_FILE in $MIGRATIONS; do
    FILENAME=$(basename "$MIGRATION_FILE")
    
    # Check if migration has already been executed
    ALREADY_EXECUTED=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM schema_migrations WHERE filename = '$FILENAME';")
    
    if [ "$ALREADY_EXECUTED" -gt 0 ]; then
        echo -e "${YELLOW}⊘ Skipping $FILENAME (already executed)${NC}"
        ((SKIPPED++))
        continue
    fi
    
    echo "Executing $FILENAME..."
    
    # Calculate checksum of migration file
    if command -v sha256sum &> /dev/null; then
        CHECKSUM=$(sha256sum "$MIGRATION_FILE" | cut -d' ' -f1)
    elif command -v shasum &> /dev/null; then
        CHECKSUM=$(shasum -a 256 "$MIGRATION_FILE" | cut -d' ' -f1)
    else
        CHECKSUM="unavailable"
    fi
    
    # Execute migration and measure time
    START_TIME=$(date +%s%N)
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$MIGRATION_FILE" > /dev/null 2>&1
    RESULT=$?
    
    END_TIME=$(date +%s%N)
    EXECUTION_TIME_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    
    if [ $RESULT -eq 0 ]; then
        # Record successful migration
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
INSERT INTO schema_migrations (filename, checksum, execution_time_ms)
VALUES ('$FILENAME', '$CHECKSUM', $EXECUTION_TIME_MS);
EOF
        echo -e "${GREEN}✓ Successfully executed $FILENAME (${EXECUTION_TIME_MS}ms)${NC}"
        ((EXECUTED++))
    else
        echo -e "${RED}✗ Failed to execute $FILENAME${NC}"
        ((FAILED++))
        
        # Stop on first failure
        echo ""
        echo -e "${RED}Migration failed. Stopping execution.${NC}"
        echo "Please fix the error and run migrations again."
        break
    fi
done

echo ""
echo "========================================"
echo "Migration Summary"
echo "========================================"
echo -e "${GREEN}Executed: $EXECUTED${NC}"
echo -e "${YELLOW}Skipped:  $SKIPPED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed:   $FAILED${NC}"
else
    echo -e "${GREEN}Failed:   $FAILED${NC}"
fi

# Show current migration status
echo ""
echo "Current migration status:"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
    "SELECT filename, executed_at FROM schema_migrations ORDER BY executed_at DESC LIMIT 5;"

# Unset PGPASSWORD
unset PGPASSWORD

if [ $FAILED -gt 0 ]; then
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All migrations completed successfully!${NC}"