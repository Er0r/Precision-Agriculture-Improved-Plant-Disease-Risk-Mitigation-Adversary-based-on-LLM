"""
Database Connection Utilities for Precision Agriculture System

This module provides utilities for connecting to PostgreSQL database
and handling database operations consistently across the application.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from django.conf import settings
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """
    PostgreSQL database connection manager
    """
    
    def __init__(self):
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None
    
    def connect(self):
        """
        Establish connection to PostgreSQL database
        """
        try:
            # Try to use Django settings first
            if hasattr(settings, 'DATABASES') and 'default' in settings.DATABASES:
                db_config = settings.DATABASES['default']
                if db_config['ENGINE'] == 'django.db.backends.postgresql':
                    self.connection = psycopg2.connect(
                        host=db_config.get('HOST', 'localhost'),
                        port=db_config.get('PORT', 5432),
                        database=db_config['NAME'],
                        user=db_config['USER'],
                        password=db_config['PASSWORD'],
                        cursor_factory=RealDictCursor
                    )
                else:
                    logger.warning("Django is configured to use SQLite, not PostgreSQL")
                    return False
            else:
                # Fallback to environment variables
                self.connection = psycopg2.connect(
                    host=os.getenv('POSTGRES_HOST', 'localhost'),
                    port=os.getenv('POSTGRES_PORT', 5432),
                    database=os.getenv('POSTGRES_DB', 'precision_agriculture'),
                    user=os.getenv('POSTGRES_USER', 'postgres'),
                    password=os.getenv('POSTGRES_PASSWORD', ''),
                    cursor_factory=RealDictCursor
                )
            
            self.cursor = self.connection.cursor()
            logger.info("Connected to PostgreSQL database successfully")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Union[List[Any], int, None]:
        """
        Execute a query and return results
        """
        try:
            if not self.cursor:
                if not self.connect():
                    return None
            
            if self.cursor:  # Type guard
                self.cursor.execute(query, params)
                
                # For SELECT queries, return results
                if query.strip().upper().startswith('SELECT'):
                    return self.cursor.fetchall()
                else:
                    # For INSERT/UPDATE/DELETE, commit and return affected rows
                    if self.connection:
                        self.connection.commit()
                    return self.cursor.rowcount
                
        except psycopg2.Error as e:
            logger.error(f"Database query error: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def close(self):
        """
        Close database connection
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed")
        except psycopg2.Error as e:
            logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_analysis_results_with_prevention():
    """
    Get analysis results that have prevention strategies
    """
    query = """
    SELECT ar.*, ci.crop_type, ci.original_filename
    FROM analysis_data ar
    JOIN analysis_cropimage ci ON ar.image_id = ci.id
    WHERE ar.prevention_strategies IS NOT NULL 
    AND ar.prevention_strategies != '[]'
    """
    
    with DatabaseConnection() as db:
        return db.execute_query(query)


def test_connection():
    """
    Test database connection
    """
    with DatabaseConnection() as db:
        if db.connection:
            result = db.execute_query("SELECT COUNT(*) as count FROM analysis_data")
            if result and isinstance(result, list) and len(result) > 0:
                count_value = result[0].get('count', 0) if isinstance(result[0], dict) else 0
                print(f"Database connection successful. Found {count_value} analysis results.")
                return True
        
        print("Database connection failed.")
        return False


if __name__ == '__main__':
    # Test the connection
    test_connection()
