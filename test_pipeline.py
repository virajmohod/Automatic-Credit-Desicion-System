import requests
import json

# The URL where your FastAPI app is running
BASE_URL = "http://127.0.0.1:8000/decide"

# Sample test cases
applicants = [
    {
        "name": "High Earners (Auto-Approve)",
        "data": {
            "income": 120000, "monthly_debt": 1000, "total_credit_limit": 50000,
            "used_credit": 2000, "loan_amount": 5000, "missed_payments_2yr": 0
        }
    },
    {
        "name": "High Debt (Manual Review)",
        "data": {
            "income": 45000, "monthly_debt": 2000, "total_credit_limit": 10000,
            "used_credit": 7000, "loan_amount": 15000, "missed_payments_2yr": 1
        }
    },
    {
        "name": "High Risk (Reject)",
        "data": {
            "income": 30000, "monthly_debt": 2500, "total_credit_limit": 5000,
            "used_credit": 4500, "loan_amount": 20000, "missed_payments_2yr": 4
        }
    }
]

def run_tests():
    print("🚀 Starting ACDS Pipeline Integration Test...\n")
    for person in applicants:
        response = requests.post(BASE_URL, json=person["data"])
        result = response.json()
        
        print(f"Scenario: {person['name']}")
        print(f"Result: {result['decision']} (Prob: {result['probability_of_default']})")
        print("-" * 30)

if __name__ == "__main__":
    run_tests()