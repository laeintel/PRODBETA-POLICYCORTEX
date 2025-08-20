"""
Apply ML database migration to PostgreSQL
"""
import psycopg2
from psycopg2 import sql
import sys
import os
from pathlib import Path

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'policycortex',
    'user': 'postgres',
    'password': 'postgres'
}

def apply_migration(migration_file):
    """Apply SQL migration file to database"""
    
    # Read migration file
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    conn = None
    cursor = None
    
    try:
        # Connect to database
        print(f"Connecting to PostgreSQL at {DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}")
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = False  # Use transaction
        cursor = conn.cursor()
        
        # Execute migration
        print(f"Applying migration from {migration_file}")
        cursor.execute(migration_sql)
        
        # Commit transaction
        conn.commit()
        print("Migration applied successfully!")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'ml_%'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"\nCreated {len(tables)} ML tables:")
        for table in tables:
            print(f"  - {table[0]}")
            
        # Count indexes
        cursor.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE tablename LIKE 'ml_%'
            AND schemaname = 'public';
        """)
        
        index_count = cursor.fetchone()[0]
        print(f"\nCreated {index_count} indexes")
        
        # Count views
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'v_%';
        """)
        
        view_count = cursor.fetchone()[0]
        print(f"Created {view_count} views")
        
        return True
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    # Get migration file path
    script_dir = Path(__file__).parent
    migration_file = script_dir / 'create_ml_tables_fixed.sql'
    
    if not migration_file.exists():
        print(f"Migration file not found: {migration_file}")
        sys.exit(1)
    
    # Apply migration
    success = apply_migration(migration_file)
    
    if success:
        print("\n✓ Database migration completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Database migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()