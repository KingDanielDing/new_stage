import requests
import json
import io

# Define the API endpoint
url = "http://127.0.0.1:5000/get-state"

# Create the payload
payload = {
    "Code": "pdd"  # Change this to the desired code name for the chart
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

