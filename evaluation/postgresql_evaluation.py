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

# Import sentiment analysis modules (with fallback if not available)
try:
    from evaluation.sentiment.vader_sentiment import VaderSentimentAnalyzer
    from evaluation.sentiment.metrics import calculate_sentiment_metrics, get_sentiment_summary
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("Warning: Sentiment analysis not available. Install vaderSentiment to enable.")
    SENTIMENT_AVAILABLE = False

# Import database connection utilities
from database.connection import DatabaseConnection


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


def evaluate_text_sentiment(text_fields: dict) -> dict:
    """
    Evaluate sentiment for given text fields using VADER with agricultural domain adjustments
    
    Args:
        text_fields: Dictionary with field names as keys and text content as values
        
    Returns:
        Dictionary with sentiment metrics for each field
    """
    if not SENTIMENT_AVAILABLE:
        return {}
    
    try:
        analyzer = VaderSentimentAnalyzer()
        sentiment_results = {}
        
        for field_name, text in text_fields.items():
            if not text or not text.strip():
                continue
                
            # Analyze sentiment
            sentiment_data = analyzer.analyze_text(text)
            metrics = calculate_sentiment_metrics(sentiment_data)
            
            # Add field-specific prefix to metrics
            field_prefix = f"{field_name}_sentiment"
            sentiment_results.update({
                f"{field_prefix}_positive": metrics['positive_score'],
                f"{field_prefix}_negative": metrics['negative_score'],
                f"{field_prefix}_neutral": metrics['neutral_score'],
                f"{field_prefix}_compound": metrics['compound_score'],
                f"{field_prefix}_domain_adjusted": metrics['domain_adjusted_compound'],
                f"{field_prefix}_classification": metrics['sentiment_classification'],
                f"{field_prefix}_confidence": metrics['confidence_level'],
                f"{field_prefix}_polarity_strength": metrics.get('domain_polarity_strength', 0.0),
                f"{field_prefix}_domain_terms": metrics['total_domain_terms'],
                f"{field_prefix}_positive_terms": metrics['domain_positive_terms'],
                f"{field_prefix}_negative_terms": metrics['domain_negative_terms']
            })
        
        return sentiment_results
        
    except Exception as e:
        print(f"Error in sentiment evaluation: {e}")
        return {}


def generate_sentiment_report(data: list) -> dict:
    """
    Generate comprehensive sentiment analysis report
    
    Args:
        data: List of evaluation records with sentiment metrics
        
    Returns:
        Dictionary with sentiment report
    """
    if not SENTIMENT_AVAILABLE:
        return {'error': 'Sentiment analysis not available'}
    
    # Extract sentiment data for different fields
    prevention_sentiments = []
    recommendations_sentiments = []
    prevention_classifications = []
    recommendations_classifications = []
    
    for record in data:
        # Prevention strategies sentiment
        if 'prevention_sentiment_compound' in record:
            try:
                prevention_sentiments.append(float(record['prevention_sentiment_compound']))
                prevention_classifications.append(record.get('prevention_sentiment_classification', 'Unknown'))
            except (ValueError, TypeError):
                pass
        
        # Recommendations sentiment
        if 'recommendations_sentiment_compound' in record:
            try:
                recommendations_sentiments.append(float(record['recommendations_sentiment_compound']))
                recommendations_classifications.append(record.get('recommendations_sentiment_classification', 'Unknown'))
            except (ValueError, TypeError):
                pass
    
    # Calculate statistics
    stats = {
        'prevention_strategies': {
            'count': len(prevention_sentiments),
            'average_compound': sum(prevention_sentiments) / len(prevention_sentiments) if prevention_sentiments else 0,
            'classification_distribution': {}
        },
        'recommendations': {
            'count': len(recommendations_sentiments),
            'average_compound': sum(recommendations_sentiments) / len(recommendations_sentiments) if recommendations_sentiments else 0,
            'classification_distribution': {}
        }
    }
    
    # Count sentiment classifications
    for classification in prevention_classifications:
        stats['prevention_strategies']['classification_distribution'][classification] = \
            stats['prevention_strategies']['classification_distribution'].get(classification, 0) + 1
    
    for classification in recommendations_classifications:
        stats['recommendations']['classification_distribution'][classification] = \
            stats['recommendations']['classification_distribution'].get(classification, 0) + 1
    
    # Generate insights
    insights = []
    recommendations = []
    
    # Prevention strategies insights
    if stats.get('prevention_strategies', {}).get('count', 0) > 0:
        prev_avg = stats['prevention_strategies']['average_compound']
        prev_dist = stats['prevention_strategies']['classification_distribution']
        
        if prev_avg > 0.1:
            insights.append({
                'field': 'Prevention Strategies',
                'sentiment': 'Positive',
                'average_score': prev_avg,
                'interpretation': 'Generally positive tone in prevention strategies'
            })
        elif prev_avg < -0.1:
            insights.append({
                'field': 'Prevention Strategies',
                'sentiment': 'Negative',
                'average_score': prev_avg,
                'interpretation': 'Generally negative tone in prevention strategies'
            })
        else:
            insights.append({
                'field': 'Prevention Strategies',
                'sentiment': 'Neutral',
                'average_score': prev_avg,
                'interpretation': 'Neutral tone in prevention strategies'
            })
        
        # Check for concerning patterns
        negative_count = prev_dist.get('Negative', 0)
        total_count = stats['prevention_strategies']['count']
        if negative_count / total_count > 0.3:
            recommendations.append("Consider rephrasing prevention strategies with more positive language")
    
    # Recommendations insights
    if stats.get('recommendations', {}).get('count', 0) > 0:
        rec_avg = stats['recommendations']['average_compound']
        
        if rec_avg > 0.1:
            insights.append({
                'field': 'Recommendations',
                'sentiment': 'Positive',
                'average_score': rec_avg,
                'interpretation': 'Generally positive tone in recommendations'
            })
        elif rec_avg < -0.1:
            insights.append({
                'field': 'Recommendations',
                'sentiment': 'Negative',
                'average_score': rec_avg,
                'interpretation': 'Generally negative tone in recommendations'
            })
        else:
            insights.append({
                'field': 'Recommendations',
                'sentiment': 'Neutral',
                'average_score': rec_avg,
                'interpretation': 'Neutral tone in recommendations'
            })
    
    # General recommendations
    if len([r for r in data if r.get('prevention_sentiment_confidence') == 'Low']) > len(data) * 0.2:
        recommendations.append("Improve sentiment confidence by using more decisive language")
    
    report = {
        'statistics': stats,
        'insights': insights,
        'recommendations': recommendations,
        'summary': {
            'total_analyzed': len(data),
            'sentiment_available': SENTIMENT_AVAILABLE,
            'analysis_date': datetime.now().isoformat()
        }
    }
    
    return report


def perform_evaluation():
    """
    Main evaluation function using organized modules including sentiment analysis
    """
    print("Starting comprehensive PostgreSQL evaluation...")
    
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
    
    # 4. Perform sentiment evaluation for each record
    if SENTIMENT_AVAILABLE:
        print("Calculating sentiment metrics...")
        for record in data:
            text_fields = {
                'prevention': record['prevention_strategies'],
                'recommendations': record['recommendations']
            }
            
            sentiment_results = evaluate_text_sentiment(text_fields)
            record.update(sentiment_results)
    else:
        print("Skipping sentiment analysis (vaderSentiment not available)")
    
    # 5. Generate evaluation reports
    print("Generating evaluation reports...")
    clarity_report = generate_clarity_report(data)
    similarity_report = generate_similarity_report(data)
    sentiment_report = generate_sentiment_report(data) if SENTIMENT_AVAILABLE else {}
    
    # 6. Save results
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
    
    # Save sentiment report
    if sentiment_report:
        sentiment_report_path = output_dir / f'sentiment_report_{timestamp}.json'
        with open(sentiment_report_path, 'w', encoding='utf-8') as f:
            json.dump(sentiment_report, f, indent=2, ensure_ascii=False)
        print(f"Sentiment report saved to: {sentiment_report_path}")
    
    # 7. Print evaluation summary
    print_evaluation_summary(data, clarity_report, similarity_report, sentiment_report)
    
    return data


def print_evaluation_summary(data: list, clarity_report: dict, similarity_report: dict, sentiment_report: dict):
    """
    Print comprehensive evaluation summary including sentiment analysis
    """
    print(f"\n=== COMPREHENSIVE EVALUATION SUMMARY ===")
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
    
    # Sentiment insights
    if sentiment_report and 'insights' in sentiment_report:
        print(f"\n--- SENTIMENT INSIGHTS ---")
        for insight in sentiment_report['insights']:
            print(f"{insight['field']}: {insight['sentiment']} (score: {insight['average_score']:.3f})")
            print(f"  → {insight['interpretation']}")
    
    # Combined recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    for rec in clarity_report.get('recommendations', []):
        print(f"• [Clarity] {rec}")
    for rec in similarity_report.get('consistency_analysis', {}).get('recommendations', []):
        print(f"• [Similarity] {rec}")
    if sentiment_report and 'recommendations' in sentiment_report:
        for rec in sentiment_report['recommendations']:
            print(f"• [Sentiment] {rec}")


if __name__ == '__main__':
    try:
        results = perform_evaluation()
        print(f"\nEvaluation completed successfully! Processed {len(results)} records.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
