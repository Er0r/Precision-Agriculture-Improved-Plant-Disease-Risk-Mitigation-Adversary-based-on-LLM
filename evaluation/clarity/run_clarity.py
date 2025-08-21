"""Script to compute readability metrics for backend/analysis_results.csv

Produces: evaluation/clarity/results_with_readability.csv

Usage: run this with the project's python environment (it uses only stdlib).
"""
import csv
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `evaluation` package imports work
# when running this script from inside the evaluation directory.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.clarity.metrics import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    smog_index,
    gunning_fog_index,
    get_domain_analysis
)
from evaluation.clarity.domain_glossary import get_domain_coverage

BASE = os.path.dirname(__file__)
# Input lives in the project-level backend folder
INPUT_CSV = os.path.join(PROJECT_ROOT, "backend", "analysis_results.csv")
# Output should be the evaluation/clarity directory under project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "clarity")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results_with_readability.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELD_TO_EVALUATE = "Treatment Optimization"

with open(INPUT_CSV, newline='', encoding='utf-8') as inf:
    reader = csv.DictReader(inf)
    rows = list(reader)

if not rows:
    print("No rows found in input CSV:", INPUT_CSV)
    raise SystemExit(1)

# prepare new fieldnames
fieldnames = list(rows[0].keys()) + [
    "FK_Grade",
    "FK_Grade_Domain_Aware",
    "Flesch_Reading_Ease",
    "Flesch_Reading_Ease_Domain_Aware",
    "SMOG_Index",
    "SMOG_Index_Domain_Aware",
    "Gunning_Fog_Index",
    "Gunning_Fog_Index_Domain_Aware",
    "Domain_Coverage_Percent",
    "Domain_Terms_Count",
    "Complex_Words_Excluded"
]

with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as outf:
    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        text = r.get(FIELD_TO_EVALUATE) or ""
        # compute metrics
        try:
            # Traditional metrics (penalize all complex words)
            r["FK_Grade"] = flesch_kincaid_grade(text, use_domain_exclusion=False)
            r["Flesch_Reading_Ease"] = flesch_reading_ease(text, use_domain_exclusion=False)
            r["SMOG_Index"] = smog_index(text, use_domain_exclusion=False)
            r["Gunning_Fog_Index"] = gunning_fog_index(text, use_domain_exclusion=False)
            
            # Domain-aware metrics (exclude domain terms from complexity penalty)
            r["FK_Grade_Domain_Aware"] = flesch_kincaid_grade(text, use_domain_exclusion=True)
            r["Flesch_Reading_Ease_Domain_Aware"] = flesch_reading_ease(text, use_domain_exclusion=True)
            r["SMOG_Index_Domain_Aware"] = smog_index(text, use_domain_exclusion=True)
            r["Gunning_Fog_Index_Domain_Aware"] = gunning_fog_index(text, use_domain_exclusion=True)
            
            # Domain analysis
            domain_analysis = get_domain_analysis(text)
            r["Domain_Coverage_Percent"] = get_domain_coverage(text)
            r["Domain_Terms_Count"] = domain_analysis['domain_terms']
            r["Complex_Words_Excluded"] = domain_analysis['domain_terms_excluded']
            
        except Exception as e:
            # Set default values for all metrics on error
            r["FK_Grade"] = r["FK_Grade_Domain_Aware"] = "ERROR"
            r["Flesch_Reading_Ease"] = r["Flesch_Reading_Ease_Domain_Aware"] = "ERROR"
            r["SMOG_Index"] = r["SMOG_Index_Domain_Aware"] = "ERROR"
            r["Gunning_Fog_Index"] = r["Gunning_Fog_Index_Domain_Aware"] = "ERROR"
            r["Domain_Coverage_Percent"] = r["Domain_Terms_Count"] = r["Complex_Words_Excluded"] = "ERROR"
            print("Error computing metrics for row", r.get("Test ID"), e)
        writer.writerow(r)

print("Wrote readability results to:", OUTPUT_CSV)
