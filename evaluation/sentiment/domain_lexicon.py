"""
Domain-Specific Sentiment Lexicon for Agricultural Analysis

This module provides agriculture-specific sentiment terms and adjustments
to enhance VADER sentiment analysis for plant disease and agricultural text.

Uses word-boundary regex matching and negation detection for robust analysis.
"""

import re
import math

# Positive agricultural terms that indicate good outcomes
AGRICULTURAL_POSITIVE_TERMS = {
    # Health and growth terms
    'healthy', 'thriving', 'robust', 'vigorous', 'strong', 'resilient',
    'flourishing', 'productive', 'fertile', 'lush', 'abundant',
    
    # Treatment success terms
    'effective', 'successful', 'improved', 'enhanced', 'optimal', 'beneficial',
    'protective', 'preventive', 'curative', 'healing', 'recovery', 'restored',
    'controlled', 'managed', 'eliminated', 'reduced', 'minimized',
    
    # Quality terms
    'excellent', 'superior', 'premium', 'high-quality', 'pure', 'clean',
    'fresh', 'organic', 'natural', 'sustainable',
    
    # Resistance and protection
    'resistant', 'immune', 'protected', 'defended', 'safe', 'secure',
    'tolerance', 'tolerant', 'adapted', 'stable',
    
    # Yield and productivity
    'increased', 'boosted', 'maximized', 'optimized', 'profitable',
    'efficient', 'high-yield', 'improved-yield'
}

# Negative agricultural terms that indicate problems or poor outcomes
AGRICULTURAL_NEGATIVE_TERMS = {
    # Disease and damage terms
    'diseased', 'infected', 'contaminated', 'damaged', 'destroyed', 'ruined',
    'wilted', 'withered', 'yellowing', 'browning', 'rotting', 'decaying',
    'stressed', 'stunted', 'deformed', 'distorted',
    
    # Severity terms
    'severe', 'critical', 'serious', 'major', 'significant', 'extensive',
    'widespread', 'devastating', 'catastrophic', 'aggressive', 'virulent',
    
    # Failure and loss terms
    'failure', 'failed', 'unsuccessful', 'ineffective', 'poor', 'bad',
    'decline', 'deterioration', 'degradation', 'loss', 'losses',
    'decreased', 'diminished', 'compromised',
    
    # Vulnerability terms
    'susceptible', 'vulnerable', 'weak', 'fragile', 'unstable',
    'sensitive', 'prone', 'at-risk',
    
    # Negative conditions
    'drought', 'flooding', 'frost', 'heat-stress', 'nutrient-deficiency',
    'toxic', 'harmful', 'dangerous', 'threatening'
}

# Neutral technical terms that should not heavily influence sentiment
AGRICULTURAL_NEUTRAL_TERMS = {
    # Scientific terms
    'analysis', 'assessment', 'evaluation', 'measurement', 'observation',
    'detection', 'identification', 'classification', 'diagnosis',
    'monitoring', 'surveillance', 'inspection', 'examination',
    
    # Technical processes
    'application', 'treatment', 'management', 'control', 'prevention',
    'cultivation', 'irrigation', 'fertilization', 'harvesting',
    'processing', 'storage', 'handling',
    
    # Measurement units and quantities
    'percentage', 'concentration', 'dosage', 'frequency', 'duration',
    'temperature', 'humidity', 'moisture', 'density', 'volume'
}

# Multi-word phrases (compiled separately)
AGRICULTURAL_POSITIVE_PHRASES = {
    'high quality', 'high-quality', 'disease free', 'disease-free',
    'pest free', 'pest-free', 'well maintained', 'well-maintained',
    'properly managed', 'highly effective'
}

AGRICULTURAL_NEGATIVE_PHRASES = {
    'crop failure', 'yield loss', 'severe damage', 'widespread infection',
    'poor quality', 'badly damaged', 'heavily infected', 'completely destroyed'
}

# Negation words for detection
NEGATION_WORDS = {
    'not', 'no', 'never', 'without', 'lack', 'lacking', 'absence', 'absent',
    'none', 'neither', 'nor', 'nothing', 'nowhere', 'nobody', 'hardly',
    'barely', 'scarcely', 'rarely', 'seldom'
}

# Compiled regex patterns for efficient matching
_positive_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in AGRICULTURAL_POSITIVE_TERMS]
_negative_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in AGRICULTURAL_NEGATIVE_TERMS]
_neutral_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in AGRICULTURAL_NEUTRAL_TERMS]
_positive_phrase_patterns = [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in AGRICULTURAL_POSITIVE_PHRASES]
_negative_phrase_patterns = [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in AGRICULTURAL_NEGATIVE_PHRASES]
_negation_patterns = [re.compile(r'\b' + re.escape(neg) + r'\b', re.IGNORECASE) for neg in NEGATION_WORDS]


def _find_domain_terms_with_context(text: str) -> dict:
    """
    Find domain terms using proper word boundary matching and negation detection.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with found terms, their positions, and negation status
    """
    tokens = text.lower().split()
    
    positive_terms = []
    negative_terms = []
    neutral_terms = []
    
    # Check for phrase matches first (they take precedence)
    for pattern in _positive_phrase_patterns:
        for match in pattern.finditer(text):
            start_pos = len(text[:match.start()].split())
            # Check for negation in 3-token window before the phrase
            negated = _is_negated(tokens, start_pos, window=3)
            positive_terms.append({
                'term': match.group(),
                'position': start_pos,
                'negated': negated
            })
    
    for pattern in _negative_phrase_patterns:
        for match in pattern.finditer(text):
            start_pos = len(text[:match.start()].split())
            negated = _is_negated(tokens, start_pos, window=3)
            negative_terms.append({
                'term': match.group(),
                'position': start_pos,
                'negated': negated
            })
    
    # Check for single word matches
    for i, token in enumerate(tokens):
        # Positive terms
        for pattern in _positive_patterns:
            if pattern.search(token):
                negated = _is_negated(tokens, i, window=3)
                positive_terms.append({
                    'term': pattern.pattern.strip(r'\b').replace('\\', ''),
                    'position': i,
                    'negated': negated
                })
                break
        
        # Negative terms  
        for pattern in _negative_patterns:
            if pattern.search(token):
                negated = _is_negated(tokens, i, window=3)
                negative_terms.append({
                    'term': pattern.pattern.strip(r'\b').replace('\\', ''),
                    'position': i,
                    'negated': negated
                })
                break
        
        # Neutral terms
        for pattern in _neutral_patterns:
            if pattern.search(token):
                neutral_terms.append({
                    'term': pattern.pattern.strip(r'\b').replace('\\', ''),
                    'position': i,
                    'negated': False  # Don't negate neutral terms
                })
                break
    
    return {
        'positive_terms': positive_terms,
        'negative_terms': negative_terms,
        'neutral_terms': neutral_terms
    }


def _is_negated(tokens: list, position: int, window: int = 3) -> bool:
    """
    Check if a term at given position is negated by words in the preceding window.
    
    Args:
        tokens: List of tokens
        position: Position of the term to check
        window: Number of tokens to look back
        
    Returns:
        True if the term is negated
    """
    start = max(0, position - window)
    preceding_tokens = tokens[start:position]
    
    for token in preceding_tokens:
        for pattern in _negation_patterns:
            if pattern.search(token):
                return True
    return False



def get_domain_sentiment_adjustment(text: str, base_compound_score: float) -> dict:
    """
    Calculate domain-specific sentiment adjustments using robust word-boundary matching,
    negation detection, and length-normalized scaling.
    
    Args:
        text: The text to analyze
        base_compound_score: Original VADER compound score
        
    Returns:
        Dictionary containing adjustment details and new score
    """
    if not text.strip():
        return {
            'original_score': base_compound_score,
            'adjusted_score': base_compound_score,
            'positive_terms_count': 0,
            'negative_terms_count': 0,
            'neutral_terms_count': 0,
            'positive_terms_found': [],
            'negative_terms_found': [],
            'neutral_terms_found': [],
            'adjustment_delta': 0.0,
            'negation_detected': False
        }
    
    # Find domain terms with negation detection
    terms_data = _find_domain_terms_with_context(text)
    
    # Count effective terms (accounting for negation)
    positive_effective = 0
    negative_effective = 0
    neutral_count = len(terms_data['neutral_terms'])
    
    positive_terms_found = []
    negative_terms_found = []
    neutral_terms_found = [term['term'] for term in terms_data['neutral_terms']]
    
    negation_detected = False
    
    # Process positive terms
    for term_data in terms_data['positive_terms']:
        if term_data['negated']:
            # Negated positive becomes negative contribution
            negative_effective += 1
            negation_detected = True
            negative_terms_found.append(f"NOT {term_data['term']}")
        else:
            positive_effective += 1
            positive_terms_found.append(term_data['term'])
    
    # Process negative terms  
    for term_data in terms_data['negative_terms']:
        if term_data['negated']:
            # Negated negative becomes positive contribution  
            positive_effective += 1
            negation_detected = True
            positive_terms_found.append(f"NOT {term_data['term']}")
        else:
            negative_effective += 1
            negative_terms_found.append(term_data['term'])
    
    # Calculate length-normalized adjustment with saturation
    word_count = len(text.split())
    
    # Base strength per term (smaller than original 0.2 to prevent overwhelming)
    base_strength = 0.15
    
    # Length normalization using sqrt to prevent long texts from dominating
    length_factor = 1.0 / math.sqrt(max(1, word_count / 10))
    
    # Calculate raw deltas
    positive_delta = positive_effective * base_strength * length_factor
    negative_delta = negative_effective * base_strength * length_factor
    
    # Net delta with saturation (tanh keeps it bounded)
    net_delta = positive_delta - negative_delta
    saturated_delta = math.tanh(net_delta * 2) * 0.2  # Scale to ±0.2 max
    
    # Apply neutral damping (proper interpolation toward zero)
    neutral_ratio = neutral_count / max(1, word_count)
    if neutral_ratio > 0.3:  # High technical content
        damping_factor = min(0.5, neutral_ratio * 0.5)
        # Interpolate toward zero rather than multiply
        saturated_delta = saturated_delta * (1 - damping_factor)
    
    # Calculate final adjusted score with clamping
    adjusted_score = base_compound_score + saturated_delta
    adjusted_score = max(-1.0, min(1.0, adjusted_score))
    
    return {
        'original_score': base_compound_score,
        'adjusted_score': adjusted_score,
        'positive_terms_count': positive_effective,
        'negative_terms_count': negative_effective,
        'neutral_terms_count': neutral_count,
        'positive_terms_found': positive_terms_found,
        'negative_terms_found': negative_terms_found,
        'neutral_terms_found': neutral_terms_found,
        'adjustment_delta': saturated_delta,
        'length_factor': length_factor,
        'neutral_ratio': neutral_ratio,
        'negation_detected': negation_detected,
        'raw_positive_delta': positive_delta,
        'raw_negative_delta': negative_delta
    }


def get_domain_terms_in_text(text: str) -> dict:
    """
    Extract and count domain-specific terms found in the text using robust matching.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with found terms by category
    """
    if not text.strip():
        return {
            'positive_terms': [],
            'negative_terms': [],
            'neutral_terms': [],
            'total_domain_terms': 0
        }
    
    terms_data = _find_domain_terms_with_context(text)
    
    # Extract just the term names for backward compatibility
    positive_terms = []
    negative_terms = []
    
    # Handle negation properly in the output
    for term_data in terms_data['positive_terms']:
        if term_data['negated']:
            negative_terms.append(f"NOT {term_data['term']}")
        else:
            positive_terms.append(term_data['term'])
    
    for term_data in terms_data['negative_terms']:
        if term_data['negated']:
            positive_terms.append(f"NOT {term_data['term']}")
        else:
            negative_terms.append(term_data['term'])
    
    neutral_terms = [term['term'] for term in terms_data['neutral_terms']]
    
    return {
        'positive_terms': list(set(positive_terms)),
        'negative_terms': list(set(negative_terms)),
        'neutral_terms': list(set(neutral_terms)),
        'total_domain_terms': len(set(positive_terms + negative_terms + neutral_terms))
    }
