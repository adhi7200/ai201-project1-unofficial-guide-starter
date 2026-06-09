import requests

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://www.reddit.com/r/utdallas.json?limit=100"
r = requests.get(url, headers=headers)
data = r.json()