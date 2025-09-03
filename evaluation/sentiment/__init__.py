"""
Sentiment Analysis Module for Agricultural Text Evaluation

This module provides sentiment analysis capabilities specifically designed
for agricultural and plant disease analysis text using VADER sentiment analysis
with domain-specific enhancements.

Modules:
- vader_sentiment: Core VADER sentiment analysis functionality
- domain_lexicon: Agriculture-specific sentiment terms and adjustments
- metrics: Sentiment metrics computation and classification
- run_sentiment: Main execution script for sentiment evaluation
"""

# Import modules conditionally to avoid import errors during package loading
try:
    from .vader_sentiment import VaderSentimentAnalyzer
    from .domain_lexicon import get_domain_sentiment_adjustment, AGRICULTURAL_POSITIVE_TERMS, AGRICULTURAL_NEGATIVE_TERMS
    from .metrics import calculate_sentiment_metrics, classify_sentiment, get_confidence_level
    
    __all__ = [
        'VaderSentimentAnalyzer',
        'get_domain_sentiment_adjustment',
        'AGRICULTURAL_POSITIVE_TERMS',
        'AGRICULTURAL_NEGATIVE_TERMS',
        'calculate_sentiment_metrics',
        'classify_sentiment',
        'get_confidence_level'
    ]
except ImportError:
    # If dependencies are not available, provide empty __all__
    __all__ = []
