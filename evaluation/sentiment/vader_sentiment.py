"""
VADER Sentiment Analysis for Agricultural Text

This module provides the core VADER sentiment analysis functionality
enhanced with agricultural domain knowledge.
"""

import re
from typing import Dict, Any, Optional
from .domain_lexicon import get_domain_sentiment_adjustment, get_domain_terms_in_text

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    # Fallback if VADER is not installed
    SentimentIntensityAnalyzer = None


class VaderSentimentAnalyzer:
    """
    Enhanced VADER sentiment analyzer for agricultural text analysis.
    """
    
    def __init__(self):
        """Initialize the VADER sentiment analyzer."""
        if SentimentIntensityAnalyzer is None:
            raise ImportError(
                "vaderSentiment is required. Install it with: pip install vaderSentiment"
            )
        
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on agricultural text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing all sentiment analysis results
        """
        if not text or not text.strip():
            return self._get_empty_result()
        
        # Get base VADER scores
        vader_scores = self.analyzer.polarity_scores(text)
        
        # Get domain-specific adjustments
        domain_adjustment = get_domain_sentiment_adjustment(
            text, vader_scores['compound']
        )
        
        # Get domain terms found in text
        domain_terms = get_domain_terms_in_text(text)
        
        # Prepare comprehensive result
        result = {
            # Basic VADER scores
            'positive_score': vader_scores['pos'],
            'negative_score': vader_scores['neg'],
            'neutral_score': vader_scores['neu'],
            'compound_score': vader_scores['compound'],
            
            # Domain-adjusted scores
            'domain_adjusted_compound': domain_adjustment['adjusted_score'],
            'domain_positive_terms_count': domain_adjustment['positive_terms_count'],
            'domain_negative_terms_count': domain_adjustment['negative_terms_count'],
            'domain_neutral_terms_count': domain_adjustment['neutral_terms_count'],
            
            # Domain terms details
            'positive_terms_found': domain_terms['positive_terms'],
            'negative_terms_found': domain_terms['negative_terms'],
            'neutral_terms_found': domain_terms['neutral_terms'],
            'total_domain_terms': domain_terms['total_domain_terms'],
            
            # Adjustment details
            'adjustment_delta': domain_adjustment['adjustment_delta'],
            'length_factor': domain_adjustment.get('length_factor', 1.0),
            'neutral_ratio': domain_adjustment.get('neutral_ratio', 0.0),
            'negation_detected': domain_adjustment.get('negation_detected', False),
            
            # Text statistics
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': self._count_sentences(text)
        }
        
        return result
    
    def get_basic_scores(self, text: str) -> Dict[str, float]:
        """
        Get basic VADER sentiment scores only.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with basic VADER scores
        """
        if not text or not text.strip():
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def get_domain_adjusted_score(self, text: str) -> float:
        """
        Get domain-adjusted compound sentiment score.
        
        Args:
            text: The text to analyze
            
        Returns:
            Domain-adjusted compound sentiment score (-1 to 1)
        """
        if not text or not text.strip():
            return 0.0
        
        base_scores = self.analyzer.polarity_scores(text)
        adjustment = get_domain_sentiment_adjustment(text, base_scores['compound'])
        return adjustment['adjusted_score']
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Get default result for empty or invalid text."""
        return {
            'positive_score': 0.0,
            'negative_score': 0.0,
            'neutral_score': 1.0,
            'compound_score': 0.0,
            'domain_adjusted_compound': 0.0,
            'domain_positive_terms_count': 0,
            'domain_negative_terms_count': 0,
            'domain_neutral_terms_count': 0,
            'positive_terms_found': [],
            'negative_terms_found': [],
            'neutral_terms_found': [],
            'total_domain_terms': 0,
            'positive_adjustment': 0.0,
            'negative_adjustment': 0.0,
            'neutral_damping': 0.0,
            'text_length': 0,
            'word_count': 0,
            'sentence_count': 0
        }
    
    def _count_sentences(self, text: str) -> int:
        """
        Count sentences in text using simple heuristics.
        
        Args:
            text: The text to analyze
            
        Returns:
            Number of sentences
        """
        if not text:
            return 0
        
        # Simple sentence counting using punctuation
        sentence_endings = re.findall(r'[.!?]+', text)
        if sentence_endings:
            return max(1, len(sentence_endings))
        
        # Fallback: count by line breaks
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return max(1, len(lines))


def create_analyzer() -> VaderSentimentAnalyzer:
    """
    Factory function to create a VaderSentimentAnalyzer instance.
    
    Returns:
        Configured VaderSentimentAnalyzer instance
    """
    return VaderSentimentAnalyzer()


def analyze_sentiment_quick(text: str) -> Dict[str, float]:
    """
    Quick sentiment analysis function for basic use cases.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with basic sentiment scores
    """
    analyzer = create_analyzer()
    return analyzer.get_basic_scores(text)
