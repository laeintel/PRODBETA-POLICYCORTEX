"""
Verify ML database schema is correctly created
"""
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'policycortex',
    'user': 'postgres',
    'password': 'postgres'
}

def verify_schema():
    """Verify all ML tables, indexes, and views are created"""
    
    conn = None
    cursor = None
    
    try:
        # Connect to database
        print("=" * 60)
        print("ML DATABASE SCHEMA VERIFICATION")
        print("=" * 60)
        
        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        # 1. Check tables
        print("\n1. TABLES")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                table_name,
                (SELECT COUNT(*) FROM information_schema.columns 
                 WHERE table_name = t.table_name) as column_count
            FROM information_schema.tables t
            WHERE table_schema = 'public' 
            AND table_name LIKE 'ml_%'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"Found {len(tables)} ML tables:")
        for table in tables:
            print(f"  - {table['table_name']:<30} ({table['column_count']} columns)")
        
        # 2. Check indexes
        print("\n2. INDEXES")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                tablename,
                COUNT(*) as index_count
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename LIKE 'ml_%'
            GROUP BY tablename
            ORDER BY tablename;
        """)
        
        indexes = cursor.fetchall()
        total_indexes = sum(idx['index_count'] for idx in indexes)
        print(f"Found {total_indexes} indexes across {len(indexes)} tables:")
        for idx in indexes:
            print(f"  - {idx['tablename']:<30} ({idx['index_count']} indexes)")
        
        # 3. Check views
        print("\n3. VIEWS")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                table_name as view_name
            FROM information_schema.views
            WHERE table_schema = 'public'
            AND table_name LIKE 'v_%'
            ORDER BY table_name;
        """)
        
        views = cursor.fetchall()
        print(f"Found {len(views)} views:")
        for view in views:
            print(f"  - {view['view_name']}")
        
        # 4. Check foreign keys
        print("\n4. FOREIGN KEY CONSTRAINTS")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name LIKE 'ml_%';
        """)
        
        fks = cursor.fetchall()
        if fks:
            print(f"Found {len(fks)} foreign key constraints:")
            for fk in fks:
                print(f"  - {fk['table_name']}.{fk['column_name']} -> {fk['foreign_table_name']}.{fk['foreign_column_name']}")
        else:
            print("No foreign key constraints found.")
        
        # 5. Check triggers
        print("\n5. TRIGGERS")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                trigger_name,
                event_object_table as table_name,
                event_manipulation as trigger_event,
                action_timing
            FROM information_schema.triggers
            WHERE trigger_schema = 'public'
            AND event_object_table LIKE 'ml_%'
            ORDER BY event_object_table, trigger_name;
        """)
        
        triggers = cursor.fetchall()
        if triggers:
            print(f"Found {len(triggers)} triggers:")
            for trigger in triggers:
                print(f"  - {trigger['trigger_name']} on {trigger['table_name']} ({trigger['action_timing']} {trigger['trigger_event']})")
        else:
            print("No triggers found.")
        
        # 6. Check functions
        print("\n6. FUNCTIONS")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                routine_name
            FROM information_schema.routines
            WHERE routine_schema = 'public'
            AND routine_type = 'FUNCTION'
            ORDER BY routine_name;
        """)
        
        functions = cursor.fetchall()
        if functions:
            print(f"Found {len(functions)} functions:")
            for func in functions:
                print(f"  - {func['routine_name']}")
        else:
            print("No functions found.")
        
        # 7. Detailed table structure for key tables
        print("\n7. KEY TABLE STRUCTURES")
        print("-" * 40)
        
        key_tables = ['ml_predictions', 'ml_models', 'ml_feedback']
        
        for table_name in key_tables:
            print(f"\n{table_name}:")
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                ORDER BY ordinal_position
                LIMIT 5;
            """, (table_name,))
            
            columns = cursor.fetchall()
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f"DEFAULT {col['column_default']}" if col['column_default'] else ""
                print(f"    - {col['column_name']:<25} {col['data_type']:<20} {nullable:<10} {default}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count = cursor.fetchone()['count']
            print(f"    Row count: {count}")
        
        # 8. Check data types used
        print("\n8. DATA TYPES USED")
        print("-" * 40)
        
        cursor.execute("""
            SELECT 
                data_type,
                COUNT(*) as usage_count
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name LIKE 'ml_%'
            GROUP BY data_type
            ORDER BY usage_count DESC;
        """)
        
        data_types = cursor.fetchall()
        print("Data types distribution:")
        for dt in data_types:
            print(f"  - {dt['data_type']:<30} ({dt['usage_count']} columns)")
        
        # Summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("-" * 40)
        
        # Check required tables
        required_tables = [
            'ml_configurations',
            'ml_models',
            'ml_predictions',
            'ml_training_jobs',
            'ml_feedback',
            'ml_feature_store',
            'ml_drift_metrics'
        ]
        
        existing_tables = [t['table_name'] for t in tables]
        missing_tables = [t for t in required_tables if t not in existing_tables]
        
        if missing_tables:
            print(f"MISSING TABLES: {', '.join(missing_tables)}")
            print("Status: FAILED - Not all required tables exist")
        else:
            print("All required tables exist")
            print(f"Total ML tables: {len(tables)}")
            print(f"Total indexes: {total_indexes}")
            print(f"Total views: {len(views)}")
            print(f"Total foreign keys: {len(fks)}")
            print(f"Total triggers: {len(triggers)}")
            print("\nStatus: PASSED - Schema verification successful!")
        
        print("=" * 60)
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    verify_schema()