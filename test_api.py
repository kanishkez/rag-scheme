"""
Test script for the Government Scheme RAG API.
Demonstrates how to use the API endpoints.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_ingest():
    """Test CSV ingestion."""
    print("\nTesting CSV ingestion...")
    try:
        payload = {
            "csv_path": "updated_data.csv",
            "force_rebuild": True
        }
        response = requests.post(f"{BASE_URL}/ingest", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_query(question: str, k: int = 3):
    """Test query endpoint."""
    print(f"\nTesting query: '{question}'")
    try:
        payload = {
            "question": question,
            "k": k
        }
        response = requests.post(f"{BASE_URL}/query", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total_matches']} matches:")
            for i, match in enumerate(data['matches'], 1):
                print(f"\n{i}. {match['scheme']}")
                print(f"   Sector: {match['sector']}")
                print(f"   State: {match['state']}")
                print(f"   Score: {match['score']:.3f}")
                print(f"   Benefits: {match['benefits'][:100]}...")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stats():
    """Test stats endpoint."""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Government Scheme RAG API Test Suite")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("Health check failed. Make sure the server is running.")
        return
    
    # Test ingestion
    print("\n" + "=" * 50)
    if not test_ingest():
        print("Ingestion failed.")
        return
    
    # Wait a moment for processing
    print("\nWaiting for processing to complete...")
    time.sleep(2)
    
    # Test queries
    print("\n" + "=" * 50)
    test_queries = [
        "schemes for farmers",
        "financial assistance for women",
        "education schemes in Maharashtra",
        "healthcare benefits for senior citizens",
        "startup funding programs"
    ]
    
    for query in test_queries:
        test_query(query, k=2)
        print("-" * 30)
    
    # Test stats
    print("\n" + "=" * 50)
    test_stats()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    main()
