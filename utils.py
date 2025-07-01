import json
from pathlib import Path

import requests


def load_test_data(filename):
    project_directory  = Path(__file__).parent.absolute()
    test_data_path = project_directory/"testData"/filename
    with open(test_data_path) as f:
        return json.load(f)


# def get_llm_response(test_data):
#     responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
#                                  json={"question": test_data["question"],
#                                        "chat_history": []
#                                        }).json()
#     print(responseDict)
#
#
# import requests


def get_llm_response(test_data):
    try:
        response = requests.post(
            "https://rahulshettyacademy.com/rag-llm/ask",
            json={
                "question": test_data["question"],
                "chat_history": []
            },
            timeout=10  # optional, to avoid hanging tests
        )
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
        responseDict = response.json()
        print("API response:", responseDict)
        return responseDict

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError:
        print("Failed to decode JSON")

    return None  # fallback to prevent crashing your test

