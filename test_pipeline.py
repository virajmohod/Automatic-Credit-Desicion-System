import requests

BASE_URL = "http://127.0.0.1:8000/decide"

applicants = [
    {
        "name": "Low Risk Applicant",
        "data": {
            "loan_amnt": 5000,
            "installment": 150,
            "annual_inc": 100000,
            "revol_bal": 2000,
            "revol_util": 10.0,
            "acc_open_past_24mths": 0
        }
    },
    {
        "name": "High Risk Applicant",
        "data": {
            "loan_amnt": 35000,
            "installment": 1200,
            "annual_inc": 40000,
            "revol_bal": 25000,
            "revol_util": 95.0,
            "acc_open_past_24mths": 10
        }
    }
]

def run_tests():
    print("🚀 Starting ACDS Pipeline Integration Test...\n")
    for person in applicants:
        response = requests.post(BASE_URL, json=person["data"])
        
        # Debugging: If the API crashes, show the error message
        if response.status_code != 200:
            print(f"❌ Error for {person['name']}: Status {response.status_code}")
            print(f"Details: {response.text}")
            continue

        result = response.json()
        print(f"Scenario: {person['name']}")
        print(f"Decision: {result['decision']}")
        print(f"PD Score: {result['probability_of_default']}")
        if result['top_risk_factors']:
            print(f"Risk Factors: {result['top_risk_factors']}")
        print("-" * 30)

if __name__ == "__main__":
    run_tests()