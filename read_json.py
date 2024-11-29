import json

with open("/home/fit/Downloads/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# Inspect a specific part of the tokenizer data:
for key, value in tokenizer_data.items():  # Or iterate through a specific part
    print(f"Key: {key}, Value: {value}, Unicode Values: {[ord(c) for c in str(value)]}")
