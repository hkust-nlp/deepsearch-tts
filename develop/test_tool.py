import json
import argparse
import requests
import threading
import time
import random
import re
import sys
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor




# Use the same API keys and models as the original script
APPKEYS = ["1791013697637502981", "1791013312122257441"]
SELECT = 1
MODEL_NAME = ["gpt-4o-2024-08-06", "https://aigc.sankuai.com/v1/openai/native/chat/completions"]
GEN_CONFIG = {"max_tokens": 4096}

# Create lock objects


def create_chat_completion(messages, functions=None, function_call=None, model="gpt-4-turbo-eva", **kwargs):
    global APPKEYS, SELECT
    APPKEY = APPKEYS[SELECT]
    headers = {
        "Content-Type": "application/json",
        "Authorization": str(APPKEY)
    }
    json_data = {"model": model, "messages": messages, "temperature": 1.0, "top_p": 1.0, "max_tokens": 8192}
    json_data.update(kwargs)
    times = 0
    while True:
        response = requests.post(MODEL_NAME[1],
                              headers=headers,
                              json=json_data)
        try:
            res = json.loads(response.text)
            if times % 10 != 0:
                SELECT = 1 - SELECT
            result = []
            for choice in res['choices']:
                if choice['finish_reason'] == 'tool_calls':
                    result.append(choice['message'].get('tool_calls', []))
                else:
                    result.append(choice['message']['content'])
            return result
        except Exception as e:
            print(f"GPT返回值解析失败, messages={response.text}, 返回={response}")
            print(APPKEY)
            if times >= 100:
                return ["GPT4结果返回异常"]
            if times % 10 == 9:
                SELECT = 1 - SELECT
                APPKEY = APPKEYS[SELECT]
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": str(APPKEY)
                }
            else:
                pattern_milliseconds = re.compile(r'(?<=Please retry after )\d+(?= milliseconds)')
                milliseconds = pattern_milliseconds.findall(str(response.text))
                if milliseconds:
                    time.sleep(int(milliseconds[0])/1000)
            times += 1
            time.sleep(random.random())
            print(f"timeout, retrying {times} times")
            
      
      
      
def create_chat_completion_tool(messages, tools=None, tool_choice=None, functions=None, function_call=None, model="gpt-4-turbo-eva", **kwargs):
    global APPKEYS, SELECT
    APPKEY = APPKEYS[SELECT]
    headers = {
        "Content-Type": "application/json",
        "Authorization": str(APPKEY)
    }
    if tools:
        json_data = {"model": model, "messages": messages, "temperature": 1.0, "top_p": 1.0, "max_tokens": 8192, "tools": tools}
    else:
        json_data = {"model": model, "messages": messages, "temperature": 1.0, "top_p": 1.0, "max_tokens": 8192}
    json_data.update(kwargs)
    times = 0
    while True:
        response = requests.post(MODEL_NAME[1],
                              headers=headers,
                              json=json_data)
        try:
            res = json.loads(response.text)
            if times % 10 != 0:
                SELECT = 1 - SELECT
            result = []
            is_tool = False
            for choice in res['choices']:
                if choice['finish_reason'] == 'tool_calls':
                    result.extend(choice['message'].get('tool_calls', []))
                    is_tool = True
                else:
                    result.append(choice['message']['content'])
            if is_tool:
                return {'type': 'tool', 'data': result}
            else:
                return {'type': 'normal', 'data': result}
        except Exception as e:
            print(f"GPT返回值解析失败, messages={response.text}, 返回={response}")
            print(APPKEY)
            if times >= 100:
                return {"type": "error", "data": ["GPT4结果返回异常"]}
            if times % 10 == 9:
                SELECT = 1 - SELECT
                APPKEY = APPKEYS[SELECT]
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": str(APPKEY)
                }
            else:
                pattern_milliseconds = re.compile(r'(?<=Please retry after )\d+(?= milliseconds)')
                milliseconds = pattern_milliseconds.findall(str(response.text))
                if milliseconds:
                    time.sleep(int(milliseconds[0])/1000)
            times += 1
            time.sleep(random.random())
            print(f"timeout, retrying {times} times")

def parse_chat_response(resp):
    assert resp['type'] in ['tool']
    
    tool_list = []
    if resp['type'] == 'tool':
        for tool in resp['data']:
            func_name = tool['function']['name']
            try:
                func_args = json.loads(tool['function']['arguments'])
            except Exception as e:
                func_args = tool['function']['arguments']
            tool_list.append({'name': func_name, 'args': func_args})
        return tool_list


# Create messages for GPT
messages = [
        {
            "role": "user",
            "content": "某位作家，当了4年农民后第一次投稿，并于次年上了大学，毕业后曾当编辑。刚开始的作品无人问津，第一次以笔名发表文章后反响平平。曾经有作品先被封禁，后重新解禁。有2个女儿。这位作家是谁？"
        }
    ]

# messages = [
#         {
#             "role": "user",
#             "content": "你好"
#         }
#     ]

tools = [
{
    "type": "function",
    "function": {
        "name": "deep_websearch",
        "description": "A web explorer that analyzes the content of searched web pages to extract factual and relevant information based on a given search query and search intent.",
        "parameters": {
            "type": "object",
            "required": [
                "search_query",
                "search_intent"
            ],
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The search query to be used for the web search. This search query will be directly used by the web explorer to search on common search engines, so make sure it follows the standard format."
                },
                "search_intent": {
                    "type": "string",
                    "description": "The search intent to be used for the web search."
                }
            }
        }
    }
}
]


tool_choice = "auto"


  
resp = create_chat_completion_tool(messages, tools=tools, tool_choice=tool_choice, model=MODEL_NAME[0], **GEN_CONFIG)


if resp['type'] == 'tool':  
    a = parse_chat_response(resp)
    print(a)
else:
    print(resp['data'][0])
