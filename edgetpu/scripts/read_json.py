import json

with open("test.json") as f:
  model_json = json.load(f)
  print(model_json)
