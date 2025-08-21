"""
API Client for PostgreSQL Evaluation

This module provides a simple way to call the evaluation API endpoint
instead of running separate scripts.
"""

import requests
import json
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EvaluationAPIClient:
    """
    Client for calling the PostgreSQL evaluation API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_evaluation_data(self) -> Optional[Dict[str, Any]]:
        """
        Get evaluation data from the API endpoint
        """
        try:
            url = f"{self.base_url}/api/postgresql-evaluation/"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning("No evaluation data found (404)")
                return None
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to server at {self.base_url}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    def get_evaluation_export(self, format_type: str = 'csv') -> Optional[bytes]:
        """
        Get evaluation data export (CSV or JSON)
        """
        try:
            url = f"{self.base_url}/api/postgresql-evaluation-export/"
            params = {'format': format_type}
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Export request failed with status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Export request failed: {e}")
            return None
    
    def print_evaluation_summary(self, data: Dict[str, Any]) -> None:
        """
        Print a summary of evaluation results
        """
        if not data:
            print("No evaluation data available")
            return
        
        records = data.get('data', [])
        statistics = data.get('statistics', {})
        
        print("\n=== POSTGRESQL EVALUATION SUMMARY ===")
        print(f"Total records: {len(records)}")
        
        if statistics:
            print(f"Crop types: {statistics.get('crop_types', {})}")
            print(f"Disease types: {statistics.get('disease_count', 0)}")
            print(f"Average prevention readability: {statistics.get('avg_prevention_readability', 0):.1f}")
            print(f"Average recommendations readability: {statistics.get('avg_recommendations_readability', 0):.1f}")
            print(f"Average similarity score: {statistics.get('avg_similarity', 0):.3f}")
        
        # Show first few records
        if records:
            print("\n=== SAMPLE RECORDS ===")
            for i, record in enumerate(records[:3]):
                print(f"Record {i+1}: {record.get('disease_name', 'Unknown')} ({record.get('crop_type', 'Unknown')})")
                print(f"  Prevention readability: {record.get('prevention_Flesch_Reading_Ease', 0):.1f}")
                print(f"  Recommendations readability: {record.get('recommendations_Flesch_Reading_Ease', 0):.1f}")
                print(f"  Similarity: {record.get('prevention_recommendations_similarity', 0):.3f}")
    
    def save_results(self, data: Dict[str, Any], output_file: str) -> bool:
        """
        Save evaluation results to a file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False


def run_evaluation_via_api(base_url: str = "http://localhost:8000", output_file: Optional[str] = None) -> bool:
    """
    Run evaluation by calling the API endpoint
    """
    client = EvaluationAPIClient(base_url)
    
    print(f"Fetching evaluation data from {base_url}...")
    data = client.get_evaluation_data()
    
    if data:
        client.print_evaluation_summary(data)
        
        if output_file:
            client.save_results(data, output_file)
        
        return True
    else:
        print("Failed to get evaluation data. Make sure:")
        print("1. Django server is running: python manage.py runserver 8000")
        print("2. Database has analysis results with prevention strategies")
        print("3. Server is accessible at the specified URL")
        return False


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Run PostgreSQL evaluation via API')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL of the API server')
    parser.add_argument('--output', help='Output file for results (optional)')
    
    args = parser.parse_args()
    
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"evaluation_results_{timestamp}.json"
    
    success = run_evaluation_via_api(args.url, args.output)
    exit(0 if success else 1)
