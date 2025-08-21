"""
Domain Glossary for Agricultural and Plant Disease Analysis

This module provides a comprehensive glossary of domain-specific terms
that should be excluded from complex word counts during readability analysis.
"""

# Core agricultural terms
AGRICULTURAL_TERMS = {
    'agriculture', 'agricultural', 'agronomic', 'agronomy',
    'crop', 'crops', 'cropping', 'cultivation', 'cultivate', 'cultivated', 'cultivar', 'cultivars',
    'farm', 'farming', 'farmer', 'farmers', 'farmland',
    'field', 'fields', 'plantation', 'plantations',
    'harvest', 'harvesting', 'harvested', 'yield', 'yields',
    'soil', 'soils', 'fertility', 'fertilizer', 'fertilizers', 'fertilization',
    'irrigation', 'irrigate', 'irrigated', 'drainage', 'draining',
    'pesticide', 'pesticides', 'insecticide', 'insecticides',
    'herbicide', 'herbicides', 'fungicide', 'fungicides',
    'organic', 'biologic', 'biological', 'sustainable', 'sustainability',
    'rotation', 'rotational', 'monoculture', 'polyculture',
    'greenhouse', 'nursery', 'seedbed', 'seedling', 'seedlings'
}

# Plant disease and pathology terms
DISEASE_TERMS = {
    'disease', 'diseases', 'pathogen', 'pathogens', 'pathogenic', 'pathology',
    'infection', 'infections', 'infected', 'infectious',
    'symptom', 'symptoms', 'symptomatic', 'asymptomatic',
    'diagnosis', 'diagnostic', 'diagnostics', 'identify', 'identification',
    'fungal', 'bacterial', 'viral', 'parasitic',
    'blight', 'rust', 'mildew', 'rot', 'wilt', 'canker', 'scab',
    'lesion', 'lesions', 'necrosis', 'necrotic', 'chlorosis', 'chlorotic',
    'spore', 'spores', 'sporulation', 'conidium', 'conidia',
    'mycelium', 'mycelia', 'hyphae', 'sclerotia',
    'epidemic', 'epidemics', 'outbreak', 'outbreaks',
    'resistance', 'resistant', 'susceptible', 'susceptibility',
    'tolerance', 'tolerant', 'immunity', 'immune'
}

# Treatment and management terms
TREATMENT_TERMS = {
    'treatment', 'treatments', 'control', 'management', 'strategy', 'strategies',
    'prevention', 'preventive', 'preventative', 'prophylactic',
    'application', 'applications', 'spray', 'spraying', 'sprayed',
    'dosage', 'concentration', 'dilution', 'mixture',
    'monitoring', 'surveillance', 'scouting', 'inspection',
    'sanitation', 'hygiene', 'sterilization', 'disinfection',
    'quarantine', 'isolation', 'removal', 'pruning',
    'biocontrol', 'predator', 'predators', 'parasitoid', 'antagonist'
}

# Technical and scientific terms
TECHNICAL_TERMS = {
    'analysis', 'analytical', 'assessment', 'evaluation',
    'algorithm', 'algorithms', 'model', 'models', 'modeling',
    'classification', 'detection', 'recognition', 'prediction',
    'accuracy', 'precision', 'sensitivity', 'specificity',
    'confidence', 'probability', 'statistical', 'statistics',
    'temperature', 'humidity', 'moisture', 'precipitation',
    'phenotype', 'genotype', 'genetics', 'genomic',
    'physiology', 'physiological', 'metabolism', 'biochemical',
    'enzyme', 'protein', 'cellular', 'molecular',
    'laboratory', 'microscope', 'microscopic', 'culture'
}

# Environmental and climate terms
ENVIRONMENTAL_TERMS = {
    'environment', 'environmental', 'climate', 'climatic', 'weather',
    'season', 'seasonal', 'summer', 'winter', 'spring', 'autumn',
    'temperature', 'humidity', 'moisture', 'precipitation', 'rainfall',
    'drought', 'flooding', 'stress', 'abiotic', 'biotic',
    'ecosystem', 'biodiversity', 'habitat', 'microclimate',
    'ventilation', 'circulation', 'airflow'
}

# Measurement and quantification terms
MEASUREMENT_TERMS = {
    'measurement', 'quantification', 'percentage', 'ratio', 'proportion',
    'severity', 'incidence', 'prevalence', 'distribution',
    'concentration', 'density', 'intensity', 'frequency',
    'diameter', 'length', 'width', 'area', 'volume',
    'millimeter', 'centimeter', 'meter', 'hectare',
    'milligram', 'gram', 'kilogram', 'liter', 'milliliter'
}

# Combine all domain terms
DOMAIN_GLOSSARY = (
    AGRICULTURAL_TERMS | 
    DISEASE_TERMS | 
    TREATMENT_TERMS | 
    TECHNICAL_TERMS | 
    ENVIRONMENTAL_TERMS | 
    MEASUREMENT_TERMS
)

def is_domain_term(word: str) -> bool:
    """
    Check if a word is a domain-specific term that should be excluded
    from complex word counts.
    
    Args:
        word: The word to check (case-insensitive)
        
    Returns:
        True if the word is in the domain glossary, False otherwise
    """
    return word.lower().strip('.,!?;:"()[]{}') in DOMAIN_GLOSSARY

def get_domain_term_count(text: str) -> int:
    """
    Count the number of domain-specific terms in the text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Number of domain terms found
    """
    import re
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return sum(1 for word in words if is_domain_term(word))

def get_domain_coverage(text: str) -> float:
    """
    Calculate the percentage of text covered by domain terms.
    
    Args:
        text: The text to analyze
        
    Returns:
        Percentage of domain term coverage (0-100)
    """
    import re
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not words:
        return 0.0
    
    domain_count = sum(1 for word in words if is_domain_term(word))
    return round((domain_count / len(words)) * 100, 2)

def get_glossary_stats() -> dict:
    """
    Get statistics about the domain glossary.
    
    Returns:
        Dictionary with glossary statistics
    """
    return {
        'total_terms': len(DOMAIN_GLOSSARY),
        'agricultural_terms': len(AGRICULTURAL_TERMS),
        'disease_terms': len(DISEASE_TERMS),
        'treatment_terms': len(TREATMENT_TERMS),
        'technical_terms': len(TECHNICAL_TERMS),
        'environmental_terms': len(ENVIRONMENTAL_TERMS),
        'measurement_terms': len(MEASUREMENT_TERMS)
    }
