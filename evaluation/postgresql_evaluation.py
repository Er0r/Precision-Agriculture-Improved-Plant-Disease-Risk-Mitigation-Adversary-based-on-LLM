#!/usr/bin/env python3
"""
PostgreSQL Evaluation Script

This script connects to the PostgreSQL database, extracts analysis results
that have prevention_strategies, and performs clarity and semantic similarity
evaluations on them using organized evaluation modules.

Usage:
    python postgresql_evaluation.py

Requirements:
    - psycopg2-binary (for PostgreSQL connection)
    - Django environment configured
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Add backend directory to Python path for Django imports
BACKEND_DIR = PROJECT_ROOT / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_disease_api.settings')

import django
django.setup()

from analysis.models import AnalysisResult, CropImage  # type: ignore
from evaluation.clarity.postgresql_clarity import (
    evaluate_text_clarity,
    calculate_clarity_statistics,
    generate_clarity_report
)
from evaluation.simmetric.postgresql_simmetric import (
    calculate_field_similarities,
    calculate_similarity_statistics,
    generate_similarity_report
)


def get_prevention_strategy_data():
    """
    Extract analysis results that have prevention_strategies from PostgreSQL
    """
    # Query for records with prevention_strategies (non-empty list)
    results = AnalysisResult.objects.filter(
        prevention_strategies__isnull=False
    ).exclude(
        prevention_strategies__exact=[]
    ).select_related('image')
    
    print(f"Found {results.count()} analysis results with prevention strategies")
    
    data = []
    for result in results:
        # Convert prevention strategies list to text for evaluation
        prevention_text = " ".join(result.prevention_strategies) if result.prevention_strategies else ""
        
        # Convert recommendations list to text
        recommendations_text = " ".join(result.recommendations) if result.recommendations else ""
        
        record = {
            'id': result.pk,
            'image_id': str(result.image.id),
            'original_filename': result.image.original_filename,
            'crop_type': result.image.crop_type,
            'disease_name': result.disease_name,
            'confidence': result.confidence,
            'severity': result.severity,
            'prevention_strategies': prevention_text,
            'recommendations': recommendations_text,
            'danger_level': result.danger_level or "",
            'economic_impact': result.economic_impact or "",
            'treatment_timeline': result.treatment_timeline or "",
            'monitoring_advice': result.monitoring_advice or "",
            'analyzed_at': result.analyzed_at.isoformat(),
        }
        data.append(record)
    
    return data


def perform_evaluation():
    """
    Main evaluation function using organized modules
    """
    print("Starting PostgreSQL evaluation...")
    
    # 1. Extract data with prevention strategies
    data = get_prevention_strategy_data()
    
    if not data:
        print("No data found with prevention strategies. Exiting.")
        return []
    
    # 2. Perform clarity evaluation for each record
    print("Calculating clarity metrics...")
    for record in data:
        text_fields = {
            'prevention': record['prevention_strategies'],
            'recommendations': record['recommendations']
        }
        
        clarity_results = evaluate_text_clarity(text_fields)
        record.update(clarity_results)
    
    # 3. Perform similarity evaluation for each record
    print("Calculating similarity metrics...")
    for record in data:
        similarity_results = calculate_field_similarities(record)
        record.update(similarity_results)
    
    # 4. Generate evaluation reports
    print("Generating evaluation reports...")
    clarity_report = generate_clarity_report(data)
    similarity_report = generate_similarity_report(data)
    
    # 5. Save results
    output_dir = PROJECT_ROOT / 'evaluation' / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed data as CSV
    csv_path = output_dir / f'evaluation_results_{timestamp}.csv'
    if data:
        fieldnames = list(data[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"CSV results saved to: {csv_path}")
    
    # Save detailed data as JSON
    json_path = output_dir / f'evaluation_results_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)
    print(f"JSON results saved to: {json_path}")
    
    # Save clarity report
    clarity_report_path = output_dir / f'clarity_report_{timestamp}.json'
    with open(clarity_report_path, 'w', encoding='utf-8') as f:
        json.dump(clarity_report, f, indent=2, ensure_ascii=False)
    print(f"Clarity report saved to: {clarity_report_path}")
    
    # Save similarity report
    similarity_report_path = output_dir / f'similarity_report_{timestamp}.json'
    with open(similarity_report_path, 'w', encoding='utf-8') as f:
        json.dump(similarity_report, f, indent=2, ensure_ascii=False)
    print(f"Similarity report saved to: {similarity_report_path}")
    
    # 6. Print summary
    print_evaluation_summary(data, clarity_report, similarity_report)
    
    return data


def print_evaluation_summary(data, clarity_report, similarity_report):
    """
    Print a summary of evaluation results
    """
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Total records evaluated: {len(data)}")
    
    # Crop type distribution
    crop_types = {}
    for record in data:
        crop_type = record['crop_type']
        crop_types[crop_type] = crop_types.get(crop_type, 0) + 1
    print(f"Crop types: {dict(crop_types)}")
    
    # Disease distribution
    diseases = set(record['disease_name'] for record in data)
    print(f"Unique diseases: {len(diseases)}")
    
    # Clarity insights
    print(f"\n--- CLARITY INSIGHTS ---")
    for insight in clarity_report.get('insights', []):
        print(f"{insight['field']}: {insight['average_score']:.1f} ({insight['interpretation']})")
    
    # Similarity insights
    print(f"\n--- SIMILARITY INSIGHTS ---")
    consistency = similarity_report.get('consistency_analysis', {}).get('overall_consistency', 'Unknown')
    print(f"Overall content consistency: {consistency}")
    
    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    for rec in clarity_report.get('recommendations', []):
        print(f"• {rec}")
    for rec in similarity_report.get('consistency_analysis', {}).get('recommendations', []):
        print(f"• {rec}")


if __name__ == '__main__':
    try:
        results = perform_evaluation()
        print(f"\nEvaluation completed successfully! Processed {len(results)} records.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
