"""
Sentiment Metrics and Classification

This module provides functions to calculate sentiment metrics,
classify sentiment, and determine confidence levels for agricultural text.
"""

from typing import Dict, Any, Tuple, Optional


def calculate_sentiment_metrics(sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive sentiment metrics from sentiment analysis data.
    
    Args:
        sentiment_data: Dictionary containing sentiment analysis results
        
    Returns:
        Dictionary with calculated metrics
    """
    # Extract base scores
    positive = sentiment_data.get('positive_score', 0.0)
    negative = sentiment_data.get('negative_score', 0.0)
    neutral = sentiment_data.get('neutral_score', 1.0)
    compound = sentiment_data.get('compound_score', 0.0)
    domain_adjusted = sentiment_data.get('domain_adjusted_compound', compound)
    
    # Calculate additional metrics
    metrics = {
        # Basic scores (normalized)
        'positive_score': round(positive, 4),
        'negative_score': round(negative, 4),
        'neutral_score': round(neutral, 4),
        'compound_score': round(compound, 4),
        'domain_adjusted_compound': round(domain_adjusted, 4),
        
        # Polarity metrics
        'polarity_strength': abs(compound),
        'domain_polarity_strength': abs(domain_adjusted),
        'polarity_difference': abs(domain_adjusted - compound),
        
        # Domain influence metrics
        'domain_positive_terms': sentiment_data.get('domain_positive_terms_count', 0),
        'domain_negative_terms': sentiment_data.get('domain_negative_terms_count', 0),
        'domain_neutral_terms': sentiment_data.get('domain_neutral_terms_count', 0),
        'total_domain_terms': sentiment_data.get('total_domain_terms', 0),
        
        # Text characteristics
        'text_length': sentiment_data.get('text_length', 0),
        'word_count': sentiment_data.get('word_count', 0),
        'sentence_count': sentiment_data.get('sentence_count', 0),
        
        # Derived metrics
        'avg_words_per_sentence': _safe_divide(
            sentiment_data.get('word_count', 0),
            sentiment_data.get('sentence_count', 1)
        ),
        'domain_term_density': _safe_divide(
            sentiment_data.get('total_domain_terms', 0),
            sentiment_data.get('word_count', 1)
        )
    }
    
    # Add classification and confidence
    classification = classify_sentiment(domain_adjusted)
    confidence = get_confidence_level(domain_adjusted, sentiment_data)
    
    metrics.update({
        'sentiment_classification': classification,
        'confidence_level': confidence,
        'is_neutral': classification == 'Neutral',
        'is_positive': classification == 'Positive',
        'is_negative': classification == 'Negative'
    })
    
    return metrics


def classify_sentiment(compound_score: float, 
                      positive_threshold: float = 0.05,
                      negative_threshold: float = -0.05) -> str:
    """
    Classify sentiment based on compound score.
    
    Args:
        compound_score: The compound sentiment score (-1 to 1)
        positive_threshold: Minimum score for positive classification
        negative_threshold: Maximum score for negative classification
        
    Returns:
        Sentiment classification: 'Positive', 'Negative', or 'Neutral'
    """
    if compound_score >= positive_threshold:
        return 'Positive'
    elif compound_score <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'


def get_confidence_level(compound_score: float, 
                        sentiment_data: Dict[str, Any]) -> str:
    """
    Determine confidence level of sentiment classification.
    
    Args:
        compound_score: The compound sentiment score
        sentiment_data: Full sentiment analysis data
        
    Returns:
        Confidence level: 'High', 'Medium', or 'Low'
    """
    polarity_strength = abs(compound_score)
    word_count = sentiment_data.get('word_count', 0)
    domain_terms = sentiment_data.get('total_domain_terms', 0)
    
    # Calculate confidence factors
    strength_factor = _get_strength_factor(polarity_strength)
    length_factor = _get_length_factor(word_count)
    domain_factor = _get_domain_factor(domain_terms, word_count)
    
    # Combine factors (weighted average)
    confidence_score = (
        strength_factor * 0.5 +  # Polarity strength is most important
        length_factor * 0.3 +    # Text length matters for reliability
        domain_factor * 0.2      # Domain terms provide context
    )
    
    # Classify confidence
    if confidence_score >= 0.7:
        return 'High'
    elif confidence_score >= 0.4:
        return 'Medium'
    else:
        return 'Low'


def get_sentiment_summary(sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of sentiment analysis results.
    
    Args:
        sentiment_data: Complete sentiment analysis data
        
    Returns:
        Dictionary with summary information
    """
    metrics = calculate_sentiment_metrics(sentiment_data)
    
    summary = {
        'overall_sentiment': metrics['sentiment_classification'],
        'confidence': metrics['confidence_level'],
        'compound_score': metrics['compound_score'],
        'domain_adjusted_score': metrics['domain_adjusted_compound'],
        'polarity_strength': metrics['domain_polarity_strength'],
        'domain_influence': {
            'positive_terms': metrics['domain_positive_terms'],
            'negative_terms': metrics['domain_negative_terms'],
            'total_terms': metrics['total_domain_terms'],
            'density': metrics['domain_term_density']
        },
        'text_stats': {
            'words': metrics['word_count'],
            'sentences': metrics['sentence_count'],
            'avg_words_per_sentence': metrics['avg_words_per_sentence']
        }
    }
    
    return summary


def compare_sentiments(text1_data: Dict[str, Any], 
                      text2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare sentiment analysis results between two texts.
    
    Args:
        text1_data: Sentiment data for first text
        text2_data: Sentiment data for second text
        
    Returns:
        Dictionary with comparison results
    """
    metrics1 = calculate_sentiment_metrics(text1_data)
    metrics2 = calculate_sentiment_metrics(text2_data)
    
    comparison = {
        'sentiment_agreement': metrics1['sentiment_classification'] == metrics2['sentiment_classification'],
        'compound_difference': abs(metrics1['compound_score'] - metrics2['compound_score']),
        'domain_adjusted_difference': abs(metrics1['domain_adjusted_compound'] - metrics2['domain_adjusted_compound']),
        'polarity_strength_difference': abs(metrics1['polarity_strength'] - metrics2['polarity_strength']),
        'more_positive': 'text1' if metrics1['compound_score'] > metrics2['compound_score'] else 'text2',
        'higher_confidence': 'text1' if metrics1['confidence_level'] == 'High' else 'text2',
        'domain_terms_comparison': {
            'text1_terms': metrics1['total_domain_terms'],
            'text2_terms': metrics2['total_domain_terms'],
            'difference': abs(metrics1['total_domain_terms'] - metrics2['total_domain_terms'])
        }
    }
    
    return comparison


# Helper functions

def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning 0 if denominator is 0."""
    return round(numerator / denominator, 4) if denominator != 0 else 0.0


def _get_strength_factor(polarity_strength: float) -> float:
    """Get confidence factor based on polarity strength (0-1)."""
    if polarity_strength >= 0.6:
        return 1.0
    elif polarity_strength >= 0.3:
        return 0.7
    elif polarity_strength >= 0.1:
        return 0.5
    else:
        return 0.2


def _get_length_factor(word_count: int) -> float:
    """Get confidence factor based on text length."""
    if word_count >= 50:
        return 1.0
    elif word_count >= 20:
        return 0.8
    elif word_count >= 10:
        return 0.6
    elif word_count >= 5:
        return 0.4
    else:
        return 0.2


def _get_domain_factor(domain_terms: int, word_count: int) -> float:
    """Get confidence factor based on domain term presence."""
    if word_count == 0:
        return 0.0
    
    density = domain_terms / word_count
    if density >= 0.2:  # 20% or more domain terms
        return 1.0
    elif density >= 0.1:  # 10-20% domain terms
        return 0.8
    elif density >= 0.05:  # 5-10% domain terms
        return 0.6
    elif domain_terms > 0:  # Some domain terms present
        return 0.4
    else:  # No domain terms
        return 0.2
