import requests
# Sending a GET request to an API
response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
# Checking if the request was successful
if response.status_code == 200:
 print(response.json()) # Prints the response in JSON format
else:
 print("Failed to retrieve data")