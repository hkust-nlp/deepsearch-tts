import csv
import base64
import json
def xor_decrypt(data, key):
    """
    XOR decrypt data with a key
    """
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])
with open("/Users/zengweihao/browse-tts/browse-tts/data/xbench-deepsearch/DeepSearch.csv", mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    questions = []
    for question in reader:
        key = question["canary"]
        question["prompt"] = xor_decrypt(base64.b64decode(question["prompt"]), key).decode('utf-8')
        question["answer"] = xor_decrypt(base64.b64decode(question["answer"]), key).decode('utf-8')
        question["problem"] = question["prompt"]
        question["Question"] = question["prompt"]
        questions.append(question)

with open("/Users/zengweihao/browse-tts/browse-tts/data/xbench-deepsearch/dev.json", mode='w', encoding='utf-8') as file:
    json.dump(questions, file, ensure_ascii=False, indent=4)
print("bupt")