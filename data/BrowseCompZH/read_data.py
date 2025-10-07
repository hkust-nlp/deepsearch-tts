import json


with open("/ssddata/wzengak/browse-tts/browse-tts/data/BrowseCompZH-Sample100/all_data_random100_sample1.json", "r") as f:
    data = json.load(f)


with open("/ssddata/wzengak/browse-tts/tongyi-deepresearch/DeepResearch/inference/eval_data/browsezh_data.jsonl", "w") as w:

    for item in data:
        item["question"] = item["Question"]
        w.write(json.dumps(item) + "\n")






print(data)