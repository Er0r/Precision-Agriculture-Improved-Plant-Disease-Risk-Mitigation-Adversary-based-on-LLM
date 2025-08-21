#!/usr/bin/env python3
"""
Clarity Evaluation Module for PostgreSQL Data

This module handles all readability and clarity evaluation functions
for data extracted from PostgreSQL database.
"""

from .metrics import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    smog_index,
    gunning_fog_index,
)


def calculate_readability_metrics(text):
    """
    Calculate readability metrics for given text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Readability metrics including FK_Grade, Flesch_Reading_Ease, 
              SMOG_Index, and Gunning_Fog_Index
    """
    if not text or not text.strip():
        return {
            'FK_Grade': 0,
            'Flesch_Reading_Ease': 0,
            'SMOG_Index': 0,
            'Gunning_Fog_Index': 0,
        }
    
    try:
        metrics = {
            'FK_Grade': flesch_kincaid_grade(text),
            'Flesch_Reading_Ease': flesch_reading_ease(text),
            'SMOG_Index': smog_index(text),
            'Gunning_Fog_Index': gunning_fog_index(text),
        }
    except Exception as e:
        print(f"Error calculating readability metrics: {e}")
        metrics = {
            'FK_Grade': "ERROR",
            'Flesch_Reading_Ease': "ERROR",
            'SMOG_Index': "ERROR",
            'Gunning_Fog_Index': "ERROR",
        }
    
    return metrics


def evaluate_text_clarity(text_data):
    """
    Evaluate clarity for multiple text fields
    
    Args:
        text_data (dict): Dictionary with text fields to evaluate
        
    Returns:
        dict: Clarity evaluation results for each field
    """
    results = {}
    
    for field_name, text in text_data.items():
        if isinstance(text, str) and text.strip():
            metrics = calculate_readability_metrics(text)
            
            # Add field prefix to metric names
            for metric_name, value in metrics.items():
                results[f"{field_name}_{metric_name}"] = value
    
    return results


def calculate_clarity_statistics(data, field_prefixes=None):
    """
    Calculate summary statistics for clarity metrics
    
    Args:
        data (list): List of records with clarity metrics
        field_prefixes (list): List of field prefixes to analyze
        
    Returns:
        dict: Summary statistics for each metric
    """
    if field_prefixes is None:
        field_prefixes = ['prevention', 'recommendations']
    
    metrics = ['FK_Grade', 'Flesch_Reading_Ease', 'SMOG_Index', 'Gunning_Fog_Index']
    stats = {}
    
    for prefix in field_prefixes:
        stats[prefix] = {}
        for metric in metrics:
            field_name = f'{prefix}_{metric}'
            values = [
                record[field_name] for record in data 
                if isinstance(record.get(field_name), (int, float)) and record[field_name] != 0
            ]
            
            if values:
                stats[prefix][metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return stats


def interpret_readability_score(score, metric_type='flesch'):
    """
    Interpret readability scores with human-readable descriptions
    
    Args:
        score (float): The readability score
        metric_type (str): Type of metric ('flesch', 'fk_grade', 'smog', 'gunning_fog')
        
    Returns:
        str: Human-readable interpretation
    """
    if metric_type.lower() == 'flesch':
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8th & 9th grade)"
        elif score >= 50:
            return "Fairly Difficult (10th to 12th grade)"
        elif score >= 30:
            return "Difficult (college level)"
        else:
            return "Very Difficult (graduate level)"
    
    elif metric_type.lower() in ['fk_grade', 'smog', 'gunning_fog']:
        if score <= 6:
            return "Elementary school level"
        elif score <= 9:
            return "Middle school level"
        elif score <= 12:
            return "High school level"
        elif score <= 16:
            return "College level"
        else:
            return "Graduate level"
    
    return "Unknown"


def generate_clarity_report(data):
    """
    Generate a comprehensive clarity evaluation report
    
    Args:
        data (list): List of evaluated records
        
    Returns:
        dict: Comprehensive clarity report
    """
    stats = calculate_clarity_statistics(data)
    
    report = {
        'total_records': len(data),
        'statistics': stats,
        'insights': [],
        'recommendations': []
    }
    
    # Generate insights
    for field, field_stats in stats.items():
        if 'Flesch_Reading_Ease' in field_stats:
            avg_flesch = field_stats['Flesch_Reading_Ease']['mean']
            interpretation = interpret_readability_score(avg_flesch, 'flesch')
            
            report['insights'].append({
                'field': field,
                'metric': 'Flesch Reading Ease',
                'average_score': avg_flesch,
                'interpretation': interpretation,
                'difficulty_level': 'High' if avg_flesch < 50 else 'Medium' if avg_flesch < 70 else 'Low'
            })
    
    # Generate recommendations
    prevention_flesch = stats.get('prevention', {}).get('Flesch_Reading_Ease', {}).get('mean', 0)
    recommendations_flesch = stats.get('recommendations', {}).get('Flesch_Reading_Ease', {}).get('mean', 0)
    
    if prevention_flesch < 60:
        report['recommendations'].append(
            "Simplify prevention strategies text - currently too difficult for general audience"
        )
    
    if recommendations_flesch < 60:
        report['recommendations'].append(
            "Simplify recommendations text - currently too difficult for general audience"
        )
    
    if abs(prevention_flesch - recommendations_flesch) > 20:
        report['recommendations'].append(
            "Standardize writing complexity between prevention strategies and recommendations"
        )
    
    return report
