#!/usr/bin/env python3
"""
Simmetric (Semantic Similarity) Evaluation Module for PostgreSQL Data

This module handles all semantic similarity evaluation functions
for data extracted from PostgreSQL database.
"""

from .simple_similarity import cosine_similarity


def calculate_field_similarities(record_data):
    """
    Calculate semantic similarities between different text fields
    
    Args:
        record_data (dict): Dictionary with text fields to compare
        
    Returns:
        dict: Similarity scores between field pairs
    """
    # Extract text fields
    prevention = record_data.get('prevention_strategies', '')
    recommendations = record_data.get('recommendations', '')
    danger = record_data.get('danger_level', '')
    economic = record_data.get('economic_impact', '')
    treatment = record_data.get('treatment_timeline', '')
    monitoring = record_data.get('monitoring_advice', '')
    
    # Calculate all pairwise similarities
    similarities = {
        'prevention_recommendations_similarity': cosine_similarity(prevention, recommendations),
        'prevention_danger_similarity': cosine_similarity(prevention, danger),
        'prevention_economic_similarity': cosine_similarity(prevention, economic),
        'prevention_treatment_similarity': cosine_similarity(prevention, treatment),
        'prevention_monitoring_similarity': cosine_similarity(prevention, monitoring),
        'recommendations_treatment_similarity': cosine_similarity(recommendations, treatment),
        'recommendations_danger_similarity': cosine_similarity(recommendations, danger),
        'recommendations_economic_similarity': cosine_similarity(recommendations, economic),
        'recommendations_monitoring_similarity': cosine_similarity(recommendations, monitoring),
        'danger_economic_similarity': cosine_similarity(danger, economic),
        'danger_treatment_similarity': cosine_similarity(danger, treatment),
        'danger_monitoring_similarity': cosine_similarity(danger, monitoring),
        'economic_treatment_similarity': cosine_similarity(economic, treatment),
        'economic_monitoring_similarity': cosine_similarity(economic, monitoring),
        'treatment_monitoring_similarity': cosine_similarity(treatment, monitoring),
    }
    
    return similarities


def calculate_similarity_statistics(data, similarity_fields=None):
    """
    Calculate summary statistics for similarity metrics
    
    Args:
        data (list): List of records with similarity metrics
        similarity_fields (list): List of similarity fields to analyze
        
    Returns:
        dict: Summary statistics for each similarity metric
    """
    if similarity_fields is None:
        similarity_fields = [
            'prevention_recommendations_similarity',
            'prevention_danger_similarity', 
            'prevention_economic_similarity',
            'prevention_treatment_similarity',
            'prevention_monitoring_similarity',
            'recommendations_treatment_similarity'
        ]
    
    stats = {}
    
    for field in similarity_fields:
        values = [
            record[field] for record in data 
            if isinstance(record.get(field), (int, float))
        ]
        
        if values:
            stats[field] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values),
                'std_dev': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
    
    return stats


def interpret_similarity_score(score):
    """
    Interpret similarity scores with human-readable descriptions
    
    Args:
        score (float): Similarity score (0-1)
        
    Returns:
        str: Human-readable interpretation
    """
    if score >= 0.8:
        return "Very High Similarity"
    elif score >= 0.6:
        return "High Similarity"
    elif score >= 0.4:
        return "Moderate Similarity"
    elif score >= 0.2:
        return "Low Similarity"
    else:
        return "Very Low Similarity"


def find_similarity_outliers(data, field, threshold_type='low', threshold=0.3):
    """
    Find records with unusually high or low similarity scores
    
    Args:
        data (list): List of records with similarity metrics
        field (str): Similarity field to analyze
        threshold_type (str): 'low' or 'high' outliers
        threshold (float): Threshold value for outlier detection
        
    Returns:
        list: Records that are outliers
    """
    outliers = []
    
    for record in data:
        score = record.get(field, 0)
        if isinstance(score, (int, float)):
            if threshold_type == 'low' and score < threshold:
                outliers.append({
                    'id': record.get('id'),
                    'disease_name': record.get('disease_name'),
                    'score': score,
                    'interpretation': interpret_similarity_score(score)
                })
            elif threshold_type == 'high' and score > threshold:
                outliers.append({
                    'id': record.get('id'),
                    'disease_name': record.get('disease_name'),
                    'score': score,
                    'interpretation': interpret_similarity_score(score)
                })
    
    return outliers


def analyze_content_consistency(data):
    """
    Analyze consistency of content across different text fields
    
    Args:
        data (list): List of evaluated records
        
    Returns:
        dict: Content consistency analysis
    """
    stats = calculate_similarity_statistics(data)
    
    analysis = {
        'overall_consistency': 'Unknown',
        'field_analysis': {},
        'consistency_issues': [],
        'recommendations': []
    }
    
    # Calculate overall consistency score
    key_similarities = [
        'prevention_recommendations_similarity',
        'prevention_treatment_similarity',
        'recommendations_treatment_similarity'
    ]
    
    overall_scores = []
    for field in key_similarities:
        if field in stats:
            overall_scores.append(stats[field]['mean'])
    
    if overall_scores:
        overall_avg = sum(overall_scores) / len(overall_scores)
        if overall_avg >= 0.6:
            analysis['overall_consistency'] = 'Good'
        elif overall_avg >= 0.4:
            analysis['overall_consistency'] = 'Moderate'
        else:
            analysis['overall_consistency'] = 'Poor'
    
    # Analyze each similarity field
    for field, field_stats in stats.items():
        mean_score = field_stats['mean']
        analysis['field_analysis'][field] = {
            'average_similarity': mean_score,
            'interpretation': interpret_similarity_score(mean_score),
            'consistency_level': 'Good' if mean_score >= 0.5 else 'Poor'
        }
        
        # Identify consistency issues
        if mean_score < 0.3:
            field_readable = field.replace('_', ' ').title()
            analysis['consistency_issues'].append(
                f"Low semantic alignment in {field_readable} (avg: {mean_score:.3f})"
            )
    
    # Generate recommendations
    prevention_rec_sim = stats.get('prevention_recommendations_similarity', {}).get('mean', 0)
    if prevention_rec_sim < 0.5:
        analysis['recommendations'].append(
            "Improve alignment between prevention strategies and recommendations"
        )
    
    prevention_treatment_sim = stats.get('prevention_treatment_similarity', {}).get('mean', 0)
    if prevention_treatment_sim < 0.4:
        analysis['recommendations'].append(
            "Better coordinate prevention strategies with treatment timelines"
        )
    
    return analysis


def generate_similarity_report(data):
    """
    Generate a comprehensive similarity evaluation report
    
    Args:
        data (list): List of evaluated records
        
    Returns:
        dict: Comprehensive similarity report
    """
    stats = calculate_similarity_statistics(data)
    consistency_analysis = analyze_content_consistency(data)
    
    # Find low similarity outliers
    outliers = find_similarity_outliers(
        data, 
        'prevention_recommendations_similarity', 
        'low', 
        0.2
    )
    
    report = {
        'total_records': len(data),
        'statistics': stats,
        'consistency_analysis': consistency_analysis,
        'low_similarity_outliers': outliers[:10],  # Top 10 outliers
        'insights': [],
        'action_items': []
    }
    
    # Generate insights
    for field, field_stats in stats.items():
        mean_score = field_stats['mean']
        field_readable = field.replace('_', ' ').replace('similarity', '').strip().title()
        
        report['insights'].append({
            'field_pair': field_readable,
            'average_similarity': mean_score,
            'interpretation': interpret_similarity_score(mean_score),
            'needs_attention': mean_score < 0.4
        })
    
    # Generate action items
    if len(outliers) > 5:
        report['action_items'].append(
            f"Review {len(outliers)} records with very low prevention-recommendations similarity"
        )
    
    low_consistency_fields = [
        field for field, analysis in consistency_analysis['field_analysis'].items()
        if analysis['consistency_level'] == 'Poor'
    ]
    
    if low_consistency_fields:
        report['action_items'].append(
            f"Improve content consistency in {len(low_consistency_fields)} field pairs"
        )
    
    return report
