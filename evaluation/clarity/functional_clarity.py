import csv
import os
import re
from typing import Optional
from .metrics import flesch_kincaid_grade, get_domain_analysis
from .domain_glossary import get_domain_coverage

# Minimal Functional Clarity-only script
BASE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, 'results_with_readability.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'results_with_functional_clarity.csv')


def calculate_completeness(actual_output: str) -> float:
    """Calculate completeness based on presence of expected elements.

    Returns score between 0 and 10.
    """
    required_elements = {
        'disease_identification': ['disease', 'pathogen', 'infection', 'blight', 'rust', 'spot', 'bacterial', 'fungal'],
        'severity_level': ['mild', 'moderate', 'severe', 'high', 'low', 'critical'],
        'treatment_action': ['apply', 'spray', 'treat', 'fungicide', 'pesticide', 'remove', 'dose'],
        'timing': ['within', 'hours', 'days', 'immediately', 'weekly', 'schedule', '48 hours', '24 hours'],
        'confidence': ['confidence', 'probability', 'likely', 'certain', '%']
    }

    found_categories = 0
    total_categories = len(required_elements)

    output_lower = (actual_output or "").lower()

    for category, keywords in required_elements.items():
        if any(keyword in output_lower for keyword in keywords):
            found_categories += 1

    completeness_score = (found_categories / total_categories) * 10
    return round(completeness_score, 2)


def linguistic_score_from_fk(fk_value, domain_coverage_percent: float = 0.0) -> float:
    """
    Apply domain-aware formula: 10 - (FK_Grade - 6) with domain expertise bonus.
    
    Args:
        fk_value: Flesch-Kincaid grade level
        domain_coverage_percent: Percentage of domain terms in text
        
    Returns:
        Linguistic score (0-10) with domain expertise bonus
    """
    try:
        fk = float(fk_value)
    except Exception:
        return 0.0
    
    base_score = 10 - (fk - 6)
    base_score = max(0.0, min(10.0, base_score))
    
    # Domain expertise bonus - high domain coverage indicates appropriate technical depth
    if domain_coverage_percent >= 20:  # High domain expertise
        domain_bonus = 1.5
    elif domain_coverage_percent >= 10:  # Moderate domain expertise
        domain_bonus = 1.0
    elif domain_coverage_percent >= 5:  # Some domain expertise
        domain_bonus = 0.5
    else:
        domain_bonus = 0.0
    
    final_score = min(10.0, base_score + domain_bonus)
    return round(final_score, 2)


def simple_actionability(text: str) -> float:
    """Simple actionability: presence of action words gives higher score (0-10)."""
    if not text:
        return 0.0
    actions = ['apply', 'spray', 'treat', 'remove', 'mix', 'dilute']
    low = text.lower()
    found = any(a in low for a in actions)
    return 9.0 if found else 1.0


def compute_functional_clarity(row: dict) -> dict:
    """Compute functional clarity using domain-aware linguistic scoring."""
    treatment = row.get('Treatment Optimization') or ""
    fk = row.get('FK_Grade') or 0

    completeness = calculate_completeness(treatment)
    
    # Get domain analysis for the treatment text
    domain_analysis = get_domain_analysis(treatment)
    domain_coverage = domain_analysis['domain_coverage_percent']
    
    # Use domain-aware linguistic scoring
    linguistic = linguistic_score_from_fk(fk, domain_coverage)
    
    # Use U = completeness, Un = linguistic. 
    U = round(completeness, 2)
    Un = round(linguistic, 2)

    FC = round((U + Un) / 2.0, 2)

    return {
        'completeness': completeness,
        'linguistic': linguistic,
        'domain_coverage_percent': domain_coverage,
        'domain_terms': domain_analysis['domain_terms'],
        'complex_words_excluded': domain_analysis['domain_terms_excluded'],
        'U': U,
        'Un': Un,
        'Functional_Clarity': FC,
    }


def main():
    if not os.path.exists(INPUT_CSV):
        print('Input CSV not found:', INPUT_CSV)
        return

    with open(INPUT_CSV, newline='', encoding='utf-8') as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)

    if not rows:
        print('No rows to process in', INPUT_CSV)
        return

    out_fieldnames = list(rows[0].keys()) + [
        'completeness', 'linguistic', 'domain_coverage_percent', 
        'domain_terms', 'complex_words_excluded', 'U', 'Un', 'Functional_Clarity'
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in rows:
            scores = compute_functional_clarity(r)
            out = dict(r)
            out.update(scores)
            writer.writerow(out)

    print('Wrote functional clarity CSV to:', OUTPUT_CSV)


if __name__ == '__main__':
    main()
