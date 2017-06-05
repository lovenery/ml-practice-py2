import requests, json

url = "http://localhost:5000/api"
data = json.dumps({"sl":5.84,"sw":3.0,"pl":3.75,"pw":1.1})
r = requests.post(url, data)

print r.json()