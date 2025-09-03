"""Script to compute sentiment analysis metrics for backend/analysis_results.csv

Produces: evaluation/sentiment/results_with_sentiment.csv

Usage: run this with the project's python environment after installing vaderSentiment.
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

from evaluation.sentiment.vader_sentiment import VaderSentimentAnalyzer
from evaluation.sentiment.metrics import calculate_sentiment_metrics

BASE = os.path.dirname(__file__)
# Input lives in the project-level backend folder
INPUT_CSV = os.path.join(PROJECT_ROOT, "backend", "analysis_results.csv")
# Output should be the evaluation/sentiment directory under project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "sentiment")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results_with_sentiment.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELD_TO_EVALUATE = "Treatment Optimization"

def main():
    """Main execution function for sentiment analysis."""
    print(f"Starting sentiment analysis...")
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found: {INPUT_CSV}")
        print("Please ensure the backend/analysis_results.csv file exists.")
        return 1
    
    # Initialize sentiment analyzer
    try:
        analyzer = VaderSentimentAnalyzer()
        print("VADER sentiment analyzer initialized successfully.")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install vaderSentiment: pip install vaderSentiment")
        return 1
    except Exception as e:
        print(f"Error initializing sentiment analyzer: {e}")
        return 1
    
    # Read input CSV
    try:
        with open(INPUT_CSV, newline='', encoding='utf-8') as inf:
            reader = csv.DictReader(inf)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return 1
    
    if not rows:
        print("No rows found in input CSV:", INPUT_CSV)
        return 1
    
    print(f"Found {len(rows)} rows to process.")
    
    # Check if target field exists
    if FIELD_TO_EVALUATE not in rows[0].keys():
        print(f"Error: Field '{FIELD_TO_EVALUATE}' not found in CSV.")
        print(f"Available fields: {list(rows[0].keys())}")
        return 1
    
    # Prepare new fieldnames
    fieldnames = list(rows[0].keys()) + [
        "Positive_Score",
        "Negative_Score", 
        "Neutral_Score",
        "Compound_Score",
        "Domain_Adjusted_Compound",
        "Sentiment_Classification",
        "Confidence_Level",
        "Polarity_Strength",
        "Domain_Positive_Terms_Count",
        "Domain_Negative_Terms_Count",
        "Domain_Neutral_Terms_Count",
        "Total_Domain_Terms",
        "Domain_Term_Density",
        "Text_Word_Count",
        "Text_Sentence_Count",
        "Avg_Words_Per_Sentence",
        "Positive_Terms_Found",
        "Negative_Terms_Found",
        "Domain_Influence_Score"
    ]
    
    processed_count = 0
    error_count = 0
    
    # Process rows and write output
    try:
        with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as outf:
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(rows, 1):
                text = row.get(FIELD_TO_EVALUATE) or ""
                
                try:
                    # Perform sentiment analysis
                    sentiment_data = analyzer.analyze_text(text)
                    metrics = calculate_sentiment_metrics(sentiment_data)
                    
                    # Add sentiment metrics to row
                    row["Positive_Score"] = metrics['positive_score']
                    row["Negative_Score"] = metrics['negative_score']
                    row["Neutral_Score"] = metrics['neutral_score']
                    row["Compound_Score"] = metrics['compound_score']
                    row["Domain_Adjusted_Compound"] = metrics['domain_adjusted_compound']
                    row["Sentiment_Classification"] = metrics['sentiment_classification']
                    row["Confidence_Level"] = metrics['confidence_level']
                    row["Polarity_Strength"] = metrics.get('domain_polarity_strength', 0.0)
                    row["Domain_Positive_Terms_Count"] = metrics['domain_positive_terms']
                    row["Domain_Negative_Terms_Count"] = metrics['domain_negative_terms']
                    row["Domain_Neutral_Terms_Count"] = metrics['domain_neutral_terms']
                    row["Total_Domain_Terms"] = metrics['total_domain_terms']
                    row["Domain_Term_Density"] = metrics['domain_term_density']
                    row["Text_Word_Count"] = metrics['word_count']
                    row["Text_Sentence_Count"] = metrics['sentence_count']
                    row["Avg_Words_Per_Sentence"] = metrics['avg_words_per_sentence']
                    
                    # Join lists for CSV output
                    row["Positive_Terms_Found"] = "; ".join(sentiment_data.get('positive_terms_found', []))
                    row["Negative_Terms_Found"] = "; ".join(sentiment_data.get('negative_terms_found', []))
                    
                    # Calculate domain influence score
                    domain_influence = calculate_domain_influence_score(sentiment_data)
                    row["Domain_Influence_Score"] = domain_influence
                    
                    processed_count += 1
                    
                    if i % 10 == 0:  # Progress indicator
                        print(f"Processed {i}/{len(rows)} rows...")
                        
                except Exception as e:
                    # Set error values for all sentiment metrics
                    sentiment_fields = [
                        "Positive_Score", "Negative_Score", "Neutral_Score", "Compound_Score",
                        "Domain_Adjusted_Compound", "Sentiment_Classification", "Confidence_Level",
                        "Polarity_Strength", "Domain_Positive_Terms_Count", "Domain_Negative_Terms_Count",
                        "Domain_Neutral_Terms_Count", "Total_Domain_Terms", "Domain_Term_Density",
                        "Text_Word_Count", "Text_Sentence_Count", "Avg_Words_Per_Sentence",
                        "Positive_Terms_Found", "Negative_Terms_Found", "Domain_Influence_Score"
                    ]
                    
                    for field in sentiment_fields:
                        row[field] = "ERROR"
                    
                    error_count += 1
                    print(f"Error processing row {i} (ID: {row.get('Test ID', 'Unknown')}): {e}")
                
                writer.writerow(row)
                
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        return 1
    
    print(f"\nSentiment analysis completed!")
    print(f"Successfully processed: {processed_count} rows")
    print(f"Errors encountered: {error_count} rows")
    print(f"Results written to: {OUTPUT_CSV}")
    
    # Print summary statistics
    print_summary_statistics(OUTPUT_CSV)
    
    return 0


def calculate_domain_influence_score(sentiment_data: dict) -> float:
    """
    Calculate a score representing how much domain terms influenced the sentiment.
    
    Args:
        sentiment_data: Complete sentiment analysis data
        
    Returns:
        Domain influence score (0.0 to 1.0)
    """
    original_score = sentiment_data.get('compound_score', 0.0)
    adjusted_score = sentiment_data.get('domain_adjusted_compound', 0.0)
    
    # Calculate absolute difference
    difference = abs(adjusted_score - original_score)
    
    # Normalize to 0-1 scale (max possible difference is 2.0)
    influence_score = min(difference / 2.0, 1.0)
    
    return round(influence_score, 4)


def print_summary_statistics(output_csv: str):
    """Print summary statistics of the sentiment analysis results."""
    try:
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            print("No data to summarize.")
            return
        
        # Filter out error rows
        valid_rows = [r for r in rows if r.get('Sentiment_Classification') != 'ERROR']
        
        if not valid_rows:
            print("No valid sentiment analysis results found.")
            return
        
        # Count sentiment classifications
        sentiment_counts = {}
        confidence_counts = {}
        compound_scores = []
        domain_adjusted_scores = []
        
        for row in valid_rows:
            sentiment = row.get('Sentiment_Classification', 'Unknown')
            confidence = row.get('Confidence_Level', 'Unknown')
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            try:
                compound_scores.append(float(row.get('Compound_Score', 0)))
                domain_adjusted_scores.append(float(row.get('Domain_Adjusted_Compound', 0)))
            except (ValueError, TypeError):
                pass
        
        print("\n--- Sentiment Analysis Summary ---")
        print(f"Total valid analyses: {len(valid_rows)}")
        
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(valid_rows)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        print(f"\nConfidence Distribution:")
        for confidence, count in confidence_counts.items():
            percentage = (count / len(valid_rows)) * 100
            print(f"  {confidence}: {count} ({percentage:.1f}%)")
        
        if compound_scores:
            avg_compound = sum(compound_scores) / len(compound_scores)
            avg_domain_adjusted = sum(domain_adjusted_scores) / len(domain_adjusted_scores)
            print(f"\nAverage Compound Score: {avg_compound:.4f}")
            print(f"Average Domain-Adjusted Score: {avg_domain_adjusted:.4f}")
            print(f"Average Domain Influence: {abs(avg_domain_adjusted - avg_compound):.4f}")
        
    except Exception as e:
        print(f"Error generating summary statistics: {e}")


if __name__ == "__main__":
    sys.exit(main())
