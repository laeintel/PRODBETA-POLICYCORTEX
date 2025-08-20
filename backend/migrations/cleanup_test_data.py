"""
Clean up test data from ML tables
"""
import psycopg2

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'policycortex',
    'user': 'postgres',
    'password': 'postgres'
}

def cleanup():
    """Clean up all test data from ML tables"""
    
    conn = None
    cursor = None
    
    try:
        # Connect to database
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Delete all test data (in correct order due to foreign keys)
        tables = [
            'ml_feedback',
            'ml_drift_metrics',
            'ml_feature_store',
            'ml_predictions',
            'ml_training_jobs',
            'ml_models',
            'ml_configurations'
        ]
        
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            print(f"Cleared table: {table}")
        
        conn.commit()
        print("\nAll test data cleared successfully!")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    cleanup()