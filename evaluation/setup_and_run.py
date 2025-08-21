#!/usr/bin/env python3
"""
Setup and Run PostgreSQL Evaluation

This script sets up the evaluation environment and runs the PostgreSQL evaluation
using the organized evaluation modules.
"""

import os
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / 'backend'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))


def setup_evaluation_environment():
    """Set up the evaluation environment"""
    print("Setting up evaluation environment...")
    
    # Create evaluation results directory
    results_dir = PROJECT_ROOT / 'evaluation' / 'results'
    results_dir.mkdir(exist_ok=True)
    print(f"Created directory: {results_dir}")
    
    # Set up Django environment
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_disease_api.settings')
    
    try:
        import django
        django.setup()
        print("Django environment configured successfully")
        return True
    except Exception as e:
        print(f"Error setting up Django: {e}")
        return False


def check_database_records():
    """Check if there are records with prevention strategies in the database"""
    try:
        from analysis.models import AnalysisResult  # type: ignore
        
        total_records = AnalysisResult.objects.count()
        prevention_records = AnalysisResult.objects.filter(
            prevention_strategies__isnull=False
        ).exclude(
            prevention_strategies__exact=[]
        ).count()
        
        print(f"Total analysis records: {total_records}")
        print(f"Records with prevention strategies: {prevention_records}")
        
        if prevention_records == 0:
            print("WARNING: No records with prevention strategies found!")
            print("Make sure you have analyzed some images that generated prevention strategies.")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error checking database: {e}")
        return False


def run_evaluation():
    """Run the PostgreSQL evaluation"""
    print("Running PostgreSQL evaluation using organized modules...")
    
    try:
        from evaluation.postgresql_evaluation import perform_evaluation
        results = perform_evaluation()
        print(f"Evaluation completed successfully! Processed {len(results)} records.")
        return True
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("PostgreSQL Evaluation Setup and Run Script")
    print("=" * 50)
    
    # Step 1: Setup environment
    if not setup_evaluation_environment():
        print("Failed to set up evaluation environment")
        return 1
    
    # Step 2: Check database records
    if not check_database_records():
        print("Database check failed")
        return 1
    
    # Step 3: Run evaluation
    if not run_evaluation():
        print("Evaluation failed")
        return 1
    
    print("\n" + "=" * 50)
    print("Setup and evaluation completed successfully!")
    print("\nNext steps:")
    print("1. Check results in evaluation/results/ directory")
    print("2. Start Django server: python manage.py runserver")
    print("3. View API: http://localhost:8000/api/postgresql-evaluation/")
    print("4. Access frontend: http://localhost:3001/postgresql-evaluation")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
