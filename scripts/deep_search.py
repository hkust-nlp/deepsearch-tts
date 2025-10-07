import os
import json
import time
import re
import asyncio
import random
from typing import Dict, List, Optional, Set, Union, Counter, Tuple
import argparse
import requests
import threading
from collections import defaultdict

# Import search functionality
from search.bing_search import (
    bing_web_search_async,
    bing_web_search_async_pro,
    google_web_search_async_pro,
    extract_relevant_info,
    extract_relevant_info_pro,
    fetch_page_content_async,
    fetch_page_content_turbo,
    extract_snippet_with_context
)
invalid_search_queries = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]

# Import prompt generators and evaluation functions
from prompts.prompts import (
    get_search_intent_instruction,
    get_deep_web_explorer_instruction,
    get_web_page_reader_instruction,
    get_click_intent_instruction
)
from evaluate.evaluate import extract_answer_fn

# Import API client
from openai import AsyncOpenAI

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

# Error indicators for content fetching
error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]


# Use the same API keys and models as the original script


def create_chat_completion_tool(messages, tools=None, tool_choice=None, functions=None, function_call=None, model="gpt-4-turbo-eva", **kwargs):
    global APPKEYS, SELECT
    APPKEY = APPKEYS[SELECT]
    headers = {
        "Content-Type": "application/json",
        "Authorization": str(APPKEY)
    }
    if tools:
        json_data = {"model": model, "messages": messages, "temperature": 1.0, "tools": tools}
    else:
        json_data = {"model": model, "messages": messages, "temperature": 1.0}
    
    # For o3 models, exclude top_p and use max_completion_tokens
    if "o3" in model.lower():
        json_data["max_completion_tokens"] = 8192
    else:
        json_data["top_p"] = 1.0
        json_data["max_tokens"] = 8192
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
                print("---------", choice['finish_reason'])
                if choice['finish_reason'] == 'tool_calls':
                    result.extend(choice['message'].get('tool_calls', []))
                    is_tool = True
                else:
                    result.append(choice['message']['content'])
            if is_tool:
                return {'type': 'tool', 'data': result, "call_messages": res['choices'][0]['message']}
            else:
                return {'type': 'normal', 'data': result, "call_messages": res['choices'][0]['message']}
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
            tool_call_id = tool['id']
            try:
                func_args = json.loads(tool['function']['arguments'])
            except Exception as e:
                func_args = tool['function']['arguments']
            tool_list.append({'name': func_name, 'args': func_args, 'tool_call_id': tool_call_id})
        return tool_list

# Create dummy tokenizers when using custom API
class DummyTokenizer:
    def __init__(self):
        self.eos_token = "[EOS]"
        
    def apply_chat_template(self, messages, *args, **kwargs):
        # Just return the message content for the first message when using custom API
        if messages and isinstance(messages, list) and len(messages) > 0:
            return messages[0]["content"]
        return ""
    
tokenizer = DummyTokenizer()
aux_tokenizer = DummyTokenizer()

def extract_between(text, start_marker, end_marker, use_custom_api=False):
    """Extracts text between two markers in a string."""
    try:
        if use_custom_api:
            # When using custom API, the stop tokens might not be included
            # So we try to extract text from start_marker to either end_marker or end of text
            # Find the last occurrence of start_marker
            start_idx = text.rfind(start_marker)
            if start_idx == -1:
                return None
            
            start_idx += len(start_marker)
            end_idx = text.find(end_marker, start_idx)
            
            if end_idx == -1:  # If end marker not found, extract until end of text
                return text[start_idx:].strip()
            else:
                return text[start_idx:end_idx].strip()
        else:
            # Original implementation for non-custom API
            pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
            # Run pattern matching with timeout
            matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
            if matches:
                return matches[0][::-1].strip()
            return None
    except Exception as e:
        print(f"---Error:---\n{str(e)}")
        print(f"-------------------")
        return None

def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search results into a readable string."""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>','').replace('</b>','')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
    return formatted_documents

def get_random_key(api_key):
    """Get a random key from a comma-separated list of keys"""
    if api_key and ',' in api_key:
        keys = api_key.split(',')
        return random.choice(keys)
    return api_key

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    generate_mode: str = "chat",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    model_name: str = "QwQ-32B",
    stop: List[str] = [END_SEARCH_QUERY],
    retry_limit: int = 3,
    bad_words: List[str] = None,
    use_custom_api: bool = False,
    custom_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_counters: Optional[Counter] = None,
    args: Optional[argparse.Namespace] = None,
    use_aihubmix: bool = False,
    aihubmix_api_url: Optional[str] = None,
    aihubmix_api_keys: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    # Set default bad_words if None
    if bad_words is None and not use_custom_api:
        bad_words = [f"{END_SEARCH_RESULT}\n\n{tokenizer.eos_token}"]
    
    for attempt in range(retry_limit):
        # Select a new random API key for each retry attempt
        current_api_key = get_random_key(api_key)
        
        try:
            async with semaphore:
                # Increment API counter if provided
                if api_counters is not None:
                    # 直接比较model_name与args的值
                    # 如果model_name等于args.model_name，则是main model
                    # 如果model_name等于args.aux_model_name，则是aux model
                    if args:
                        if hasattr(args, 'model_name') and model_name == args.model_name:
                            api_counters['main_model'] += 1
                        elif hasattr(args, 'aux_model_name') and model_name == args.aux_model_name:
                            api_counters['aux_model'] += 1
                        else:
                            # 如果无法确定，则使用简单的启发式方法
                            if 'gpt-4o' in model_name.lower():
                                api_counters['aux_model'] += 1
                            else:
                                api_counters['main_model'] += 1
                    else:
                        # 没有args参数，使用简单的启发式方法
                        if 'gpt-4o' in model_name.lower():
                            api_counters['aux_model'] += 1
                        else:
                            api_counters['main_model'] += 1
                
                if generate_mode == "chat":
                    messages = [{"role": "user", "content": prompt}]
                    if use_custom_api:
                        # When using custom API, just use the prompt directly
                        formatted_prompt = prompt
                    else:
                        if 'qwq' in model_name.lower() or 'deepseek' in model_name.lower() or 'r1' in model_name.lower():
                            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        else:
                            formatted_prompt = aux_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if ('deepseek' in model_name.lower() or 'r1' in model_name.lower()) and "<think>\n" not in formatted_prompt:
                            formatted_prompt = formatted_prompt + "<think>\n"
                else:
                    formatted_prompt = prompt

                if (use_custom_api and custom_api_url) or (use_aihubmix and aihubmix_api_url):
                    
                    # Use custom API similar to CustomAPICompletionSampler
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": str(current_api_key) if current_api_key else "empty"
                    }
                    
                    json_data = {
                        "model": model_name, 
                        "messages": [{"role": "user", "content": formatted_prompt}], 
                        "temperature": temperature, 
                        "include_stop_str_in_output": True,
                        "stop": stop
                    }
                    
                    # For o3 models, exclude top_p and use max_completion_tokens
                    if "o3" in model_name.lower():
                        json_data["max_completion_tokens"] = max_tokens
                    else:
                        json_data["top_p"] = top_p
                        json_data["max_tokens"] = max_tokens
                    
                    # For extra parameters supported by the model
                    if repetition_penalty != 1.0:
                        json_data["repetition_penalty"] = repetition_penalty
                    if top_k != 1:
                        json_data["top_k"] = top_k
                    if min_p > 0:
                        json_data["min_p"] = min_p
                    
                    # Make synchronous request with asyncio to maintain compatibility
                    if use_aihubmix and aihubmix_api_url:
                        response_text = await asyncio.to_thread(
                            _make_aihubmix_api_request,
                            aihubmix_api_url,
                            headers,
                            json_data,
                            aihubmix_api_keys
                        )
                    else:
                        response_text = await asyncio.to_thread(
                            _make_custom_api_request,
                            custom_api_url,
                            headers,
                            json_data
                        )
                    #print("response_text", response_text)
                    return formatted_prompt, response_text
                else:
                    # Original AsyncOpenAI implementation
                    # For o3 models, exclude top_p and use max_completion_tokens
                    if "o3" in model_name.lower():
                        response = await client.completions.create(
                            model=model_name,
                            prompt=formatted_prompt,
                            temperature=temperature,
                            max_completion_tokens=max_tokens,
                            stop=stop,
                            extra_body={
                                'top_k': top_k,
                                'include_stop_str_in_output': True,
                                'repetition_penalty': repetition_penalty,
                                'bad_words': bad_words,
                                'min_p': min_p
                            },
                            timeout=3600,
                        )
                    else:
                        response = await client.completions.create(
                            model=model_name,
                            prompt=formatted_prompt,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            stop=stop,
                            extra_body={
                                'top_k': top_k,
                                'include_stop_str_in_output': True,
                                'repetition_penalty': repetition_penalty,
                                'bad_words': bad_words,
                                'min_p': min_p
                            },
                            timeout=3600,
                        )
                    return formatted_prompt, response.choices[0].text
        except Exception as e:
            print(f"Generate Response Error occurred with key {current_api_key}: {e}, Starting retry attempt {attempt + 1}")
            # print(prompt)
            if "maximum context length" in str(e).lower():
                # If length exceeds limit, reduce max_tokens by half
                max_tokens = max_tokens // 2
                print(f"Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""


def _make_custom_api_request(api_url, headers, json_data, max_retries=100):
    """Helper function to make custom API requests with retries"""
    # Extract the original API key from headers for possible random selection on retries
    original_auth = headers.get("Authorization", "")
    api_key = original_auth.replace("Bearer ", "") if original_auth.startswith("Bearer ") else original_auth
    
    for i in range(max_retries):
        # On retry, select a new random API key if the original key contains commas
        if i > 0 and ',' in api_key:
            current_key = get_random_key(api_key)
            headers["Authorization"] = f"Bearer {current_key}" if current_key else original_auth
            print(f"Retry {i}/{max_retries} with new key: {current_key}")
            
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=json_data,
                timeout=60
            )
            
            if response.status_code == 200:
                try:
                    res = json.loads(response.text)
                    return res['choices'][0]['message']['content']
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error parsing API response: {e}")
                    print(f"Response text: {response.text}")
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                print(f"Rate limited. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                continue
            # Handle context length exceeded errors - limit to 5 retries
            if response.status_code == 400:
                try:
                    error_data = json.loads(response.text)
                    error_message = error_data.get('error', {}).get('message', '')
                    error_code = error_data.get('error', {}).get('code', '')
                    
                    if 'context_length_exceeded' in error_code or 'maximum context length' in error_message:
                        # Only retry 5 times for context length exceeded errors
                        if i >= 4:  # This will be the 5th attempt (0-based index)
                            print(f"Context length exceeded error. Max retries (5) reached.")
                            return "Error: Failed after 5 retries due to context length exceeded."
                        print(f"Context length exceeded error. Retry {i+1}/5...")
                except Exception as e:
                    # If we can't parse the error JSON, just log and continue with normal retry logic
                    print(f"Error parsing 400 response: {e}")
                    
            # For other errors
            print(f"API request failed with status {response.status_code}: {response.text}")
            
        except Exception as e:
            print(f"Request error: {e}")
        
        # Exponential backoff
        #sleep_time = (2 ** i) + random.random()
        sleep_time = 1
        print(f"Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
    
    return "Error: Failed to get response after multiple retries."

def _make_aihubmix_api_request(api_url, headers, json_data, aihubmix_api_keys=None, max_retries=200):
    """Helper function to make AIHubMix API requests with retries
    
    Args:
        api_url: The AIHubMix API endpoint URL
        headers: Headers for the API request
        json_data: JSON data for the request
        aihubmix_api_keys: List of API keys or comma-separated string of keys for AIHubMix
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response content from the API
    """
    # Set up proxy environment variables
    # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
    
    # Determine API keys to use
    api_keys = []
    original_auth = headers.get("Authorization", "")
    original_key = original_auth.replace("Bearer ", "") if original_auth.startswith("Bearer ") else original_auth
    
    if aihubmix_api_keys:
        # If explicit keys are provided, use them
        if isinstance(aihubmix_api_keys, list):
            api_keys = aihubmix_api_keys
        elif isinstance(aihubmix_api_keys, str) and ',' in aihubmix_api_keys:
            api_keys = aihubmix_api_keys.split(',')
        else:
            api_keys = [aihubmix_api_keys]
    elif ',' in original_key:
        # Fall back to original key if no explicit keys provided
        api_keys = original_key.split(',')
    else:
        # Default key if nothing else is available
        api_keys = [original_key]
    
    # Randomly select an API key for this request
    current_api_key = random.choice(api_keys)
    headers["Authorization"] = "Bearer " + str(current_api_key)
    
    # Add provider settings for AIHubMix
    if "provider" not in json_data:
        # Determine 'only' based on model
        only_setting = []  # Default
        
        if "model" in json_data:
            if json_data["model"] == "moonshotai/kimi-k2":
                only_setting = ['moonshotai/fp8']
            elif json_data["model"] == "z-ai/glm-4.5":
                only_setting = ['z-ai/fp8']
            elif json_data["model"] == "openai/gpt-4o-mini":
                only_setting = ['azure']
                
        provider_config = {
            "ignore": [
                'deepinfra/fp4',
                'novita/fp8',
                'deepinfra/fp8'
            ],
            #"only": only_setting
        }
        
        # Only add quantizations if model is not openai/gpt-4o-mini
        if json_data.get("model") != "openai/gpt-4o-mini":
            provider_config['quantizations'] = ['fp8']
            
        json_data["provider"] = provider_config
    
    # Track retry attempts and token usage
    times = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    
    while times < max_retries:
        try:
            # Make API request
            response = requests.post(
                api_url,
                headers=headers,
                json=json_data,
                timeout=60
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    res = json.loads(response.text)
                    
                    # Extract token usage information if available
                    if "usage" in res:
                        usage = res.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        print(f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")
                    
                    # Process response
                    result = []
                    should_retry = False
                    
                    for choice in res['choices']:
                        finish_reason = choice.get('finish_reason', '')
                        print(f"Finish reason: {finish_reason}")
                        
                        if finish_reason == 'error':
                            print(f"Received error finish reason. Retrying request...")
                            should_retry = True
                            break
                        elif finish_reason is None:
                            print(f"Received None finish reason. Retrying request...")
                            should_retry = True
                            break
                        else:
                            result.append(choice['message']['content'])
                    
                    # If we should retry, continue to the next iteration
                    if should_retry:
                        times += 1
                        # Switch to a random API key for the retry
                        current_api_key = random.choice(api_keys)
                        headers["Authorization"] = "Bearer " + str(current_api_key)
                        print(f"Switched to random API key for error retry: {current_api_key}")
                        
                        # Simple backoff with some randomness
                        sleep_time = 1 + random.random()
                        print(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        continue
                    
                    # Return the content
                    return result[0]
                    
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error parsing API response: {e}")
                    print(f"Response text: {response.text}")
            
            # Handle rate limiting
            if response.status_code == 429:
                # Extract retry time if available
                pattern_milliseconds = re.compile(r'(?<=Please retry after )\d+(?= milliseconds)')
                milliseconds = pattern_milliseconds.findall(str(response.text))
                if milliseconds:
                    wait_time = int(milliseconds[0])/1000
                else:
                    wait_time = 1 + random.random()
                    
                print(f"Rate limited. Retrying after {wait_time} seconds.")
                time.sleep(wait_time)
                times += 1
                
                # Select a different random API key for the next attempt
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                print(f"Switching to a random API key: {current_api_key}")
                continue
            
            # Handle authentication errors - use a different random API key
            if response.status_code == 401:
                print("Authentication error. Trying a different API key.")
                # Remove the failed key from the list if we have multiple keys
                if len(api_keys) > 1:
                    api_keys = [key for key in api_keys if key != current_api_key]
                
                # Select a new random key
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                print(f"Switched to random API key: {current_api_key}")
                times += 1
                continue
            
            # Handle 400 errors with high risk content
            if response.status_code == 400:
                # Reduce max retries to 1/10 of original
                reduced_max_retries = max(1, max_retries // 20)
                if times >= reduced_max_retries:
                    print(f"Reached reduced retry limit ({reduced_max_retries}) for high risk content. Giving up.")
                    return "Error: Request contains high risk content that cannot be processed."
                print(f"High risk content detected. Using reduced retry limit: {reduced_max_retries}")
            # For other errors
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
            
            # Switch to a random API key after several failures
            if times % 3 == 2:  # Every 3rd attempt
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                print(f"Switched to random API key: {current_api_key}")
            
        except requests.exceptions.Timeout:
            print(f"Request timed out. Retrying {times+1}/{max_retries}...")
            # Try a different random API key on timeout
            current_api_key = random.choice(api_keys)
            headers["Authorization"] = "Bearer " + str(current_api_key)
            print(f"Switched to random API key after timeout: {current_api_key}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error. Retrying {times+1}/{max_retries}...")
            # Try a different random API key on connection error
            current_api_key = random.choice(api_keys)
            headers["Authorization"] = "Bearer " + str(current_api_key)
            print(f"Switched to random API key after connection error: {current_api_key}")
        except Exception as e:
            print(f"Request error: {e}")
        
        # Simple backoff with some randomness
        sleep_time = 1 + random.random()
        print(f"Retrying in {sleep_time:.2f} seconds... in _make_aihubmix_api_request")
        time.sleep(sleep_time)
        times += 1
    
    return "Error: Failed to get response after multiple retries."

async def generate_deep_web_explorer(
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    search_query: str,
    document: str,
    search_intent: str,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    search_cache_lock: asyncio.Lock,
    url_cache_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    model_semaphore: asyncio.Semaphore,
    bing_semaphore: asyncio.Semaphore,
    aux_model_semaphore: asyncio.Semaphore,
    api_counters: Optional[Counter] = None,
    use_aihubmix: bool = False,
    aihubmix_api_url: Optional[str] = None,
    aihubmix_api_keys: Optional[str] = None,
) -> Tuple[str, List[Dict], str]:
    """
    Generate deep web exploration with multiple search and click operations
    Returns the output, list of interaction records, and initial prompt
    """
    prompt = get_deep_web_explorer_instruction(search_query=search_query, search_intent=search_intent, search_result=document)
    output = ""
    original_prompt = ""
    total_tokens = len(prompt.split())  # Track total tokens including prompt
    MAX_TOKENS = 30000
    MAX_INTERACTIONS = 10  # Maximum combined number of searches and clicks
    clicked_urls = set()  # Track clicked URLs
    executed_search_queries = set()  # Track executed search queries
    total_interactions = 0
    finished = False
    first_generation = True

    while True:
        # Generate next response
        try:
            formatted_prompt, response = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                prompt=prompt,
                semaphore=aux_model_semaphore,
                generate_mode="chat" if first_generation else "completion",
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k_sampling,
                min_p=args.min_p,
                stop=[END_SEARCH_QUERY, END_CLICK_LINK],
                use_custom_api=args.use_custom_api,
                custom_api_url=args.custom_api_url,
                api_key=args.aux_api_key,
                use_aihubmix=use_aihubmix,
                aihubmix_api_url=aihubmix_api_url,
                aihubmix_api_keys=aihubmix_api_keys,
                api_counters=api_counters,
                args=args,
            )
        except asyncio.CancelledError:
            print(f"Response generation for web explorer was cancelled for search query: {search_query}")
            return f"Error: Response generation was cancelled for search query: {search_query}", original_prompt
        except Exception as e:
            print(f"Error generating response for web explorer: {str(e)}")
            return f"Error generating response: {str(e)}", original_prompt
        
        #print("webthinker response:", response)
        #print("webthinker formatted_prompt:", formatted_prompt)
        if first_generation:
            original_prompt = formatted_prompt
            prompt = formatted_prompt
        
        output += response.replace('</think>\n','')
        total_tokens = len(prompt.split()) + len(response.split())
        first_generation = False

        if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS:
            break

        # Check for search query - adapt for custom API
        has_search_query = False
        has_click_link = False
        
        if args.use_custom_api:
            # For custom API, check if response contains search query or click link markers
            if BEGIN_SEARCH_QUERY in response:
                # Find the last occurrence of the search query marker
                last_marker_pos = response.rfind(BEGIN_SEARCH_QUERY)
                
                # Check if this is the last special token in the output (no other special tokens after it)
                remaining_text = response[last_marker_pos + len(BEGIN_SEARCH_QUERY):]
                
                # Check if any other special tokens exist in the remaining text
                other_tokens = [
                    END_SEARCH_QUERY, 
                    BEGIN_SEARCH_RESULT, 
                    END_SEARCH_RESULT,
                    BEGIN_CLICK_LINK,
                    END_CLICK_LINK,
                    BEGIN_CLICK_RESULT,
                    END_CLICK_RESULT
                ]
                
                has_other_tokens = any(token in remaining_text for token in other_tokens)
                
                if not has_other_tokens:
                    # If search marker is at the end with no other tokens, append END_SEARCH_QUERY if needed
                    has_search_query = True
                    if END_SEARCH_QUERY not in remaining_text:
                        response += END_SEARCH_QUERY
                        output += END_SEARCH_QUERY
            
            # For click links, apply similar logic
            if BEGIN_CLICK_LINK in response:
                # Find the last occurrence of the click link marker
                last_marker_pos = response.rfind(BEGIN_CLICK_LINK)
                
                # Check if this is the last special token in the output
                remaining_text = response[last_marker_pos + len(BEGIN_CLICK_LINK):]
                
                # Check if any other special tokens exist in the remaining text
                other_tokens = [
                    END_CLICK_LINK,
                    BEGIN_SEARCH_QUERY,
                    END_SEARCH_QUERY, 
                    BEGIN_SEARCH_RESULT, 
                    END_SEARCH_RESULT,
                    BEGIN_CLICK_RESULT,
                    END_CLICK_RESULT
                ]
                
                has_other_tokens = any(token in remaining_text for token in other_tokens)
                
                if not has_other_tokens:
                    # If click marker is at the end with no other tokens, append END_CLICK_LINK if needed
                    has_click_link = True
                    if END_CLICK_LINK not in response:
                        response += END_CLICK_LINK
                        output += END_CLICK_LINK
        else:
            # For standard API, check if response ends with search query or click link markers
            has_search_query = response.rstrip().endswith(END_SEARCH_QUERY)
            has_click_link = response.rstrip().endswith(END_CLICK_LINK)

        if has_search_query:
            new_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY, use_custom_api=args.use_custom_api)
            total_interactions += 1
            if new_query is None or END_SEARCH_QUERY in new_query or len(new_query) <= 5 or new_query in invalid_search_queries:
                continue
            if new_query:
                if new_query in executed_search_queries:
                    # If search query was already executed, don't skip response generation
                    search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}\n\nOkay,"
                    output += search_result
                    prompt += output
                    total_tokens += len(search_result.split())
                    
                    # Generate response with the "already searched" message instead of skipping
                    try:
                        formatted_prompt, response = await generate_response(
                            client=client if 'qwq' in args.model_name.lower() else aux_client,
                            model_name=args.model_name if 'qwq' in args.model_name.lower() else args.aux_model_name,
                            prompt=prompt,
                            semaphore=model_semaphore if 'qwq' in args.model_name.lower() else aux_model_semaphore,
                            generate_mode="completion",
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            repetition_penalty=args.repetition_penalty,
                            top_k=args.top_k_sampling,
                            min_p=args.min_p,
                            stop=[END_SEARCH_QUERY, END_CLICK_LINK],
                            use_custom_api=args.use_custom_api,
                            custom_api_url=args.custom_api_url,
                            api_key=args.api_key if 'qwq' in args.model_name.lower() else args.aux_api_key,
                            use_aihubmix=use_aihubmix,
                            aihubmix_api_url=aihubmix_api_url,
                            aihubmix_api_keys=aihubmix_api_keys,
                            api_counters=api_counters,
                            args=args,
                        )
                    except asyncio.CancelledError:
                        print(f"Response generation was cancelled for duplicate search query: {new_query}")
                        output += "\nError: Response generation was cancelled"
                        prompt += "\nError: Response generation was cancelled"
                        continue
                    
                    output += response.replace('</think>\n', '')
                    total_tokens += len(response.split())
                    prompt += response.replace('</think>\n', '')
                    continue

                executed_search_queries.add(new_query)  # Add query to executed set
                
                # Execute search
                results = None
                # Check cache with lock
                async with search_cache_lock:
                    if new_query in search_cache:
                        results = search_cache[new_query]
                
                if results is None:
                    try:
                        if args.use_google_pro:
                            # Use Google Search Pro API
                            results = await google_web_search_async_pro(
                                new_query, 
                                api_key=args.google_pro_api_key,
                                semaphore=bing_semaphore,
                                api_counters=api_counters,
                            )
                            # Update cache with lock
                            async with search_cache_lock:
                                search_cache[new_query] = results
                        elif args.use_bing_pro:
                            # Use Bing Search Pro API
                            results = await bing_web_search_async_pro(
                                new_query, 
                                token=args.bing_pro_token, 
                                api=args.bing_pro_api,
                                semaphore=bing_semaphore,
                                api_counters=api_counters,
                            )
                            # Update cache with lock
                            async with search_cache_lock:
                                search_cache[new_query] = results
                        else:
                            # Use standard Bing Search API
                            results = await bing_web_search_async(
                                new_query, 
                                args.bing_subscription_key, 
                                args.bing_endpoint,
                                semaphore=bing_semaphore,
                                api_counters=api_counters,
                            )
                            # Update cache with lock
                            async with search_cache_lock:
                                search_cache[new_query] = results
                    except asyncio.CancelledError:
                        print(f"Search operation was cancelled for query: {new_query}")
                        search_result = f"\n{BEGIN_SEARCH_RESULT}\nError: Search operation was cancelled.\n{END_SEARCH_RESULT}\n\n"
                        output += search_result
                        prompt += output
                        continue
                    except Exception as e:
                        print(f"Error during search query '{new_query}': {e}")
                        results = {}
                print('- Searched for:', new_query)

                # Extract relevant information based on search API used
                if args.use_google_pro or args.use_bing_pro:
                    relevant_info = extract_relevant_info_pro(results)[:args.top_k]
                else:
                    relevant_info = extract_relevant_info(results)[:args.top_k]

                # Fetch all URLs in one batch
                urls_to_fetch = [doc_info['url'] for doc_info in relevant_info]
                contents = {}
                if urls_to_fetch:
                    try:
                        contents = await fetch_page_content_async(
                            urls_to_fetch,
                            use_jina=args.use_jina,
                            jina_api_key=args.jina_api_key,
                            keep_links=args.keep_links,
                            api_counters=api_counters,
                            max_concurrent=200,
                            show_progress=False,
                        )   
                    except Exception as e:
                        print(f"Error fetching URLs: {e}")

                for doc_info in relevant_info:
                    url = doc_info['url']
                    raw_content = contents.get(url, "")
                    if raw_content:
                        is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)
                    has_error = any(indicator.lower() in raw_content.lower() for indicator in error_indicators) or raw_content == "" or raw_content.startswith("Error:")
                    if has_error:
                        doc_info['page_info'] = "Can not fetch the page content."
                    else:
                        doc_info['page_info'] = raw_content

                formatted_documents = format_search_results(relevant_info)
                
                # Append search results
                search_result = f"\n{BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{END_SEARCH_RESULT}\n"
                output += search_result
                prompt += output
                total_tokens += len(search_result.split())
                
        # Check for click link
        elif has_click_link:
            url = extract_between(response, BEGIN_CLICK_LINK, END_CLICK_LINK, use_custom_api=args.use_custom_api)
            total_interactions += 1
            try:
                _, click_intent = await generate_response(
                    client=aux_client,
                    model_name=args.aux_model_name,
                    max_tokens=1000,
                    prompt=get_click_intent_instruction(output),
                    semaphore=aux_model_semaphore,
                    use_custom_api=args.use_custom_api,
                    custom_api_url=args.custom_api_url,
                    api_key=args.aux_api_key,
                    use_aihubmix=use_aihubmix,
                    aihubmix_api_url=aihubmix_api_url,
                    aihubmix_api_keys=aihubmix_api_keys,
                    api_counters=api_counters,
                    args=args,
                )
            except asyncio.CancelledError:
                print(f"Click intent generation was cancelled for URL: {url}")
                click_result = f"\n{BEGIN_CLICK_RESULT}\nError: Click intent generation was cancelled.\n{END_CLICK_RESULT}\n\n"
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
                continue

            if url and click_intent:
                if url in clicked_urls:
                    # If URL was already clicked, append message
                    click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\n\nOkay,"
                    output += click_result
                    prompt += output
                    total_tokens += len(click_result.split())
                    continue

                clicked_urls.add(url)  # Add URL to clicked set
                print(f"- Clicking on URL: {url} with intent: {click_intent}")
                
                # Fetch and process page content
                content = None
                # Check URL cache with lock
                async with url_cache_lock:
                    if url in url_cache:
                        content = url_cache[url]
                
                if content is None:
                    try:
                        content_dict = await fetch_page_content_async(
                            [url], 
                            use_jina=args.use_jina, 
                            jina_api_key=args.jina_api_key, 
                            keep_links=args.keep_links,
                            api_counters=api_counters,
                            max_concurrent=200,
                            show_progress=False,
                        )
                        content = content_dict[url]
                        
                        # Only cache content if it doesn't contain error indicators
                        has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or content == '' or content.startswith("Error:")
                        
                        if not has_error:
                            # Update cache with lock
                            async with url_cache_lock:
                                url_cache[url] = content
                    except asyncio.CancelledError:
                        print(f"Page content fetching was cancelled for URL: {url}")
                        click_result = f"\n{BEGIN_CLICK_RESULT}\nError: Page content fetching was cancelled.\n{END_CLICK_RESULT}\n\n"
                        output += click_result
                        prompt += output
                        total_tokens += len(click_result.split())
                        continue
                    except Exception as e:
                        print(f"Error fetching URL {url}: {e}")
                        content = f"Error fetching URL: {str(e)}"

                # Check if content has error indicators
                has_error = any(indicator.lower() in content.lower() for indicator in error_indicators) or content == '' or content.startswith("Error:")
                
                if has_error:
                    # If content has error, use it directly as summary
                    summary = "Unable to fetch the page content. You can try other links."
                else:
                    # Use web page reader to summarize content
                    try:
                        reader_prompt = get_web_page_reader_instruction(click_intent, content)
                        _, summary = await generate_response(
                            client=aux_client,
                            prompt=reader_prompt,
                            semaphore=aux_model_semaphore,
                            max_tokens=3600,
                            model_name=args.aux_model_name,
                            use_custom_api=args.use_custom_api,
                            custom_api_url=args.custom_api_url,
                            api_key=args.aux_api_key,
                            use_aihubmix=use_aihubmix,
                            aihubmix_api_url=aihubmix_api_url,
                            aihubmix_api_keys=aihubmix_api_keys,
                            api_counters=api_counters,
                            args=args,
                        )
                    except asyncio.CancelledError:
                        print(f"Content summary generation was cancelled for URL: {url}")
                        summary = "Error: Content summary generation was cancelled."

                # Append click results
                click_result = f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
        
        else:
            finished = True
            break

    # Add max limit message if needed
    if not finished and (total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS):
        output += f"\n{BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        prompt += output
        try:
            _, final_response = await generate_response(
                client=client if 'qwq' in args.model_name.lower() else aux_client,
                model_name=args.model_name if 'qwq' in args.model_name.lower() else args.aux_model_name,
                prompt=prompt,
                semaphore=model_semaphore if 'qwq' in args.model_name.lower() else aux_model_semaphore,
                generate_mode="completion",
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=512,
                repetition_penalty=1.2,
                top_k=args.top_k_sampling,
                min_p=args.min_p,
                use_custom_api=args.use_custom_api,
                custom_api_url=args.custom_api_url,
                api_key=args.api_key if 'qwq' in args.model_name.lower() else args.aux_api_key,
                use_aihubmix=use_aihubmix,
                aihubmix_api_url=aihubmix_api_url,
                aihubmix_api_keys=aihubmix_api_keys,
                api_counters=api_counters,
                args=args,
            )
            output += final_response
        except asyncio.CancelledError:
            print(f"Final summary generation was cancelled")
            output += "\nError: Final summary generation was cancelled."

    return output, original_prompt



class PolicyTool:
    """Tool that act as a policy maker"""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        aux_api_base_url: str = "http://localhost:8000",
        model_name: str = "QwQ-32B",
        aux_model_name: str = "Qwen2.5-32B-Instruct",
        api_key: str = "empty",
        aux_api_key: str = "empty",
        system_message: str = "You are a helpful assistant.",
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        max_search_calls: int = 15,
        use_jina: bool = False,
        jina_api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        min_p: float = 0.05,
        top_k_sampling: int = 20,
        repetition_penalty: float = 1.05,
        max_tokens: int = 81920,
        concurrent_limit: int = 32,
        model_concurrent_limit: Optional[int] = None,
        aux_model_concurrent_limit: Optional[int] = None,
        use_custom_api: bool = False,
        custom_api_url: Optional[str] = None,
        use_aihubmix: bool = False,
        aihubmix_api_url: Optional[str] = None,
        aihubmix_api_keys: Optional[str] = None,
        cache_dir: str = './cache',
        price_config_path: str = './model_config/aihubmix_price.json',
    ):
        """Initialize the PolicyTool with configuration parameters."""
        # Store configuration
        self.api_base_url = api_base_url
        self.aux_api_base_url = aux_api_base_url
        self.model_name = model_name
        self.aux_model_name = aux_model_name
        self.api_key = api_key
        self.aux_api_key = aux_api_key
        
        # Update system message with max_search_calls if it contains a placeholder
        if "{max_search_calls}" in system_message:
            self.system_message = system_message.format(max_search_calls=max_search_calls)
        else:
            self.system_message = system_message
            
        self.tools = tools
        self.tool_choice = tool_choice
        self.max_search_calls = max_search_calls
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k_sampling = top_k_sampling
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.use_custom_api = use_custom_api
        self.custom_api_url = custom_api_url
        self.use_aihubmix = use_aihubmix
        self.aihubmix_api_url = aihubmix_api_url
        self.aihubmix_api_keys = aihubmix_api_keys
        self.price_config_path = price_config_path
        
        # Set up concurrency limits
        self.model_concurrent_limit = model_concurrent_limit if model_concurrent_limit is not None else concurrent_limit
        self.aux_model_concurrent_limit = aux_model_concurrent_limit if aux_model_concurrent_limit is not None else concurrent_limit
        
        # Set up cache directory
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize API clients
        self.client = None
        self.aux_client = None
        if not use_custom_api:
            init_api_key = api_key.split(',')[0] if ',' in api_key else api_key
            init_aux_api_key = aux_api_key.split(',')[0] if ',' in aux_api_key else aux_api_key
            
            self.client = AsyncOpenAI(
                api_key=init_api_key,
                base_url=api_base_url,
            )
            self.aux_client = AsyncOpenAI(
                api_key=init_aux_api_key,
                base_url=aux_api_base_url,
            )
        
        # API counters
        self.api_counters = Counter({
            'main_model': 0,
            'aux_model': 0,
        })
        
        # Token counters
        self.token_counters = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # Load price configuration if available
        self.price_config = {}
        try:
            if os.path.exists(self.price_config_path):
                with open(self.price_config_path, 'r') as f:
                    self.price_config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load price configuration from {self.price_config_path}: {e}")

    def calculate_cost(self):
        """Calculate the cost based on token usage and pricing information."""
        if not self.price_config:
            return None
            
        # Get the model name to use for pricing
        model_for_pricing = self.model_name.lower()
        
        # Find the matching model in price config
        matching_model = None
        for model_name in self.price_config:
            if model_name.lower() in model_for_pricing or model_for_pricing in model_name.lower():
                matching_model = model_name
                break
        
        # If no exact match, use the first model as default
        if not matching_model and self.price_config:
            matching_model = next(iter(self.price_config))
            print(f"Warning: No pricing found for model {self.model_name}. Using {matching_model} pricing as default.")
        
        if matching_model:
            pricing = self.price_config[matching_model]
            prompt_price_per_k = pricing.get('prompt_tokens', 0)
            completion_price_per_k = pricing.get('completion_tokens', 0)
            
            prompt_cost = (self.token_counters['prompt_tokens'] / 1000) * prompt_price_per_k
            completion_cost = (self.token_counters['completion_tokens'] / 1000) * completion_price_per_k
            total_cost = prompt_cost + completion_cost
            
            return {
                'model': matching_model,
                'prompt_cost': prompt_cost,
                'completion_cost': completion_cost,
                'total_cost': total_cost,
                'pricing': {
                    'prompt_price_per_k': prompt_price_per_k,
                    'completion_price_per_k': completion_price_per_k
                }
            }
        
        return None

    def get_random_key(self, api_key):
        """Get a random key from a comma-separated list of keys"""
        if api_key and ',' in api_key:
            keys = api_key.split(',')
            return random.choice(keys)
        return api_key
        
    def _make_custom_api_request_tool(self, messages, tools=None, tool_choice=None, max_retries=50):
        """
        Make a custom API request with support for tools and retry logic.
        
        Args:
            messages: The messages to send to the API
            tools: Optional list of tools to include in the request
            tool_choice: Optional tool choice parameter
            max_retries: Maximum number of retry attempts
            
        Returns:
            Processed response object with type and data
        """
        # Extract the original API key for possible random selection on retries
        current_api_key = self.get_random_key(self.api_key)
        
        # Prepare headers for API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(current_api_key)
        }
        
        # Prepare request data
        json_data = {
            "model": self.model_name, 
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # For o3 models, exclude top_p and use max_completion_tokens
        if "o3" in self.model_name.lower():
            json_data["max_completion_tokens"] = self.max_tokens
        else:
            json_data["top_p"] = self.top_p
            json_data["max_tokens"] = self.max_tokens
        
        # Add tools if provided
        if tools:
            json_data["tools"] = tools
        if tool_choice:
            json_data["tool_choice"] = tool_choice
            
        # Add additional parameters if needed
        if self.repetition_penalty != 1.0:
            json_data["repetition_penalty"] = self.repetition_penalty
        if self.top_k_sampling != 1:
            json_data["top_k"] = self.top_k_sampling
        if self.min_p > 0:
            json_data["min_p"] = self.min_p
        
        # Track context length error retries separately
        context_length_retries = 0
        
        for i in range(max_retries):
            # On retry, select a new random API key if the original key contains commas
            if i > 0 and ',' in self.api_key:
                current_key = self.get_random_key(self.api_key)
                headers["Authorization"] = str(current_key)
                print(f"Retry {i}/{max_retries} with new key: {current_key}")
                
            try:
                # Make API request
                response = requests.post(
                    self.custom_api_url,
                    headers=headers,
                    json=json_data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    try:
                        res = json.loads(response.text)
                        
                        # Process response based on type
                        result = []
                        is_tool = False
                        
                        for choice in res['choices']:
                            print("---------", choice['finish_reason'])
                            if choice['finish_reason'] == 'tool_calls':
                                result.extend(choice['message'].get('tool_calls', []))
                                is_tool = True
                            else:
                                result.append(choice['message']['content'])
                        
                        # Create response object
                        if is_tool:
                            return {'type': 'tool', 'data': result, "call_messages": res['choices'][0]['message'], "raw_response": res}
                        else:
                            return {'type': 'normal', 'data': result, "call_messages": res['choices'][0]['message'], "raw_response": res}
                    except (KeyError, json.JSONDecodeError) as e:
                        print(f"Error parsing API response: {e}")
                        print(f"Response text: {response.text}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    print(f"Rate limited. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                
                # Handle context length exceeded errors - limit to 5 retries
                if response.status_code == 400:
                    try:
                        error_data = json.loads(response.text)
                        error_message = error_data.get('error', {}).get('message', '')
                        error_code = error_data.get('error', {}).get('code', '')
                        
                        if 'context_length_exceeded' in error_code or 'maximum context length' in error_message:
                            context_length_retries += 1
                            # Only retry 5 times for context length exceeded errors
                            if context_length_retries >= 5:
                                print(f"Context length exceeded error. Max retries (5) reached.")
                                return {"type": "error", "data": ["Error: Failed after 5 retries due to context length exceeded."]}
                            
                            # Reduce max_tokens by half for next attempt
                            json_data["max_tokens"] = json_data.get("max_tokens", self.max_tokens) // 2
                            print(f"Context length exceeded error. Reducing max_tokens to {json_data['max_tokens']}. Retry {context_length_retries}/5...")
                            continue
                    except Exception as e:
                        # If we can't parse the error JSON, just log and continue with normal retry logic
                        print(f"Error parsing 400 response: {e}")
                
                # Handle authentication errors
                if response.status_code == 401:
                    print("Authentication error. Trying a different API key.")
                    if ',' in self.api_key:
                        current_key = self.get_random_key(self.api_key)
                        headers["Authorization"] = str(current_key)
                        continue
                    else:
                        return {"type": "error", "data": ["Error: Authentication failed. Please check your API key."]}
                
                # For other errors
                print(f"API request failed with status {response.status_code}: {response.text}")
                
            except requests.exceptions.Timeout:
                print(f"Request timed out. Retrying {i+1}/{max_retries}...")
            except requests.exceptions.ConnectionError:
                print(f"Connection error. Retrying {i+1}/{max_retries}...")
            except Exception as e:
                print(f"Request error: {e}")
            
            # Exponential backoff with some randomness
            sleep_time = 1
            print(f"Retrying in {sleep_time:.2f} seconds... in _make_custom_api_request_tool")
            time.sleep(sleep_time)
        
        return {"type": "error", "data": ["Error: Failed to get response after multiple retries."]}
        
    def _make_aihubmix_api_request_tool(self, messages, aihubmix_api_keys=None, aihubmix_api_url=None, tools=None, tool_choice=None, max_retries=200):
        """
        Make a request to the AIHubMix API with support for tools and retry logic.
        
        Args:
            messages: The messages to send to the API
            aihubmix_api_keys: List of API keys or comma-separated string of keys for AIHubMix
            aihubmix_api_url: The AIHubMix API endpoint URL
            tools: Optional list of tools to include in the request
            tool_choice: Optional tool choice parameter
            max_retries: Maximum number of retry attempts
            
        Returns:
            Processed response object with type and data
        """
        # Set up proxy environment variables
        # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
        # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
        
        # Set default API URL if not provided
        if aihubmix_api_url is None:
            aihubmix_api_url = "https://aihubmix.com/v1/chat/completions"
        
        # Determine API keys to use
        api_keys = []
        if aihubmix_api_keys:
            # If explicit keys are provided, use them
            if isinstance(aihubmix_api_keys, list):
                api_keys = aihubmix_api_keys
            elif isinstance(aihubmix_api_keys, str) and ',' in aihubmix_api_keys:
                api_keys = aihubmix_api_keys.split(',')
            else:
                api_keys = [aihubmix_api_keys]
        elif ',' in self.api_key:
            # Fall back to self.api_key if no explicit keys provided
            api_keys = self.api_key.split(',')
        else:
            # Default key if nothing else is available
            api_keys = ["sk-7lwnQNrbElFFkHgUEbA2E12eE9944648BeDdDcB432D1C096"]
        
        # Randomly select an API key for this request
        current_api_key = random.choice(api_keys)
        
        # Prepare headers for API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + str(current_api_key)
        }
        
        # Prepare request data
        json_data = {
            "model": self.model_name, 
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # For o3 models, exclude top_p and use max_completion_tokens
        if "o3" in self.model_name.lower():
            json_data["max_completion_tokens"] = self.max_tokens
        else:
            json_data["top_p"] = self.top_p
            json_data["max_tokens"] = self.max_tokens
        
        # Determine 'only' based on model
        only_setting = []  # Default
        
        if self.model_name == "moonshotai/kimi-k2":
            #only_setting = ['moonshotai/fp8']
            only_setting = []
        elif self.model_name == "z-ai/glm-4.5":
            only_setting = ['z-ai/fp8']
        elif self.model_name == "openai/gpt-4o-mini":
            only_setting = ['azure']
            
        # Add provider settings
        json_data["provider"] = {
            "ignore": [
                'deepinfra/fp4',
                'novita/fp8',
                'deepinfra/fp8',
                'baseten/fp4'
                
            ],
            # 'quantizations': [
            #     'fp8'
            # ],
        }
        
        # Only add "only" field if only_setting is not empty
        if only_setting:
            json_data["provider"]["only"] = only_setting
        
        # Add tools if provided
        if tools:
            json_data["tools"] = tools
        if tool_choice:
            json_data["tool_choice"] = tool_choice
        
        # Track retry attempts
        times = 0
        
        while times < max_retries:
            try:
                # Make API request
                # print(json_data)
                # print(headers)
                # print(aihubmix_api_url)
                response = requests.post(
                    aihubmix_api_url,
                    headers=headers,
                    json=json_data,
                    timeout=60
                )
                
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")  # Print full response text
                
                if response.status_code == 200:
                    try:
                        res = json.loads(response.text)
                        
                        # Extract token usage information if available
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0
                        
                        if "usage" in res:
                            usage = res.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)
                            print(f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")
                            
                            # Update token counters
                            self.token_counters['prompt_tokens'] += prompt_tokens
                            self.token_counters['completion_tokens'] += completion_tokens
                            self.token_counters['total_tokens'] += total_tokens
                            
                            # Print current token totals
                            print(f"Total tokens so far: prompt={self.token_counters['prompt_tokens']}, completion={self.token_counters['completion_tokens']}, total={self.token_counters['total_tokens']}")
                            
                            # # Check if completion_tokens is 0 and retry if it is
                            # if completion_tokens == 0:
                            #     print(f"Received zero completion tokens. Retrying request...")
                            #     should_retry = True
                            #     times += 1
                            #     # Switch to a random API key for the retry
                            #     current_api_key = random.choice(api_keys)
                            #     headers["Authorization"] = "Bearer " + str(current_api_key)
                            #     print(f"Switched to random API key for zero completion tokens retry: {current_api_key}")
                                
                            #     # Simple backoff with some randomness
                            #     sleep_time = 1 + random.random()
                            #     print(f"Retrying in {sleep_time:.2f} seconds...")
                            #     time.sleep(sleep_time)
                            #     continue
                        
                        # Process response based on type
                        result = []
                        is_tool = False
                        should_retry = False
                        print("res", len(res['choices']))
                        for choice in res['choices']:
                        
                            finish_reason = choice.get('finish_reason', '')
                            print(f"Finish reason: {finish_reason}")
                            
                            if finish_reason == 'error':
                                print(f"Received error finish reason. Retrying request...")
                                should_retry = True
                                break
                            elif finish_reason == 'tool_calls' and 'tool_calls' in choice.get('message', {}):
                                result.extend(choice['message'].get('tool_calls', []))
                                is_tool = True
                            elif finish_reason == None:
                                print(f"Received None finish reason. Retrying request...")
                                should_retry = True
                                break
                            else:
                                result.append(choice['message']['content'])
                        
                        # If we should retry, continue to the next iteration
                        if should_retry:
                            times += 1
                            # Switch to a random API key for the retry
                            current_api_key = random.choice(api_keys)
                            headers["Authorization"] = "Bearer " + str(current_api_key)
                            print(f"Switched to random API key for error retry: {current_api_key}")
                            
                            # Simple backoff with some randomness
                            sleep_time = 1 + random.random()
                            print(f"Retrying in {sleep_time:.2f} seconds... in _make_aihubmix_api_request_tool")
                            time.sleep(sleep_time)
                            continue
                        
                        # Create response object
                        if is_tool:
                            return {
                                'type': 'tool', 
                                'data': result, 
                                "call_messages": res['choices'][0]['message'], 
                                "raw_response": res,
                                "token_usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": total_tokens
                                }
                            }
                        else:
                            return {
                                'type': 'normal', 
                                'data': result, 
                                "call_messages": res['choices'][0]['message'], 
                                "raw_response": res,
                                "token_usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": total_tokens
                                }
                            }
                    except (KeyError, json.JSONDecodeError) as e:
                        print(f"Error parsing API response: {e}")
                        print(f"Response text: {response.text}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    # Extract retry time if available
                    pattern_milliseconds = re.compile(r'(?<=Please retry after )\d+(?= milliseconds)')
                    milliseconds = pattern_milliseconds.findall(str(response.text))
                    if milliseconds:
                        wait_time = int(milliseconds[0])/1000
                    else:
                        wait_time = 1 + random.random()
                        
                    print(f"Rate limited. Retrying after {wait_time} seconds.")
                    time.sleep(wait_time)
                    times += 1
                    
                    # Select a different random API key for the next attempt
                    current_api_key = random.choice(api_keys)
                    headers["Authorization"] = "Bearer " + str(current_api_key)
                    print(f"Switching to a random API key: {current_api_key}")
                    continue
                
                # Handle authentication errors - use a different random API key
                if response.status_code == 401:
                    print("Authentication error. Trying a different API key.")
                    # Remove the failed key from the list if we have multiple keys
                    if len(api_keys) > 1:
                        api_keys = [key for key in api_keys if key != current_api_key]
                    
                    # Select a new random key
                    current_api_key = random.choice(api_keys)
                    headers["Authorization"] = "Bearer " + str(current_api_key)
                    print(f"Switched to random API key: {current_api_key}")
                    times += 1
                    continue
                
                # Handle 400 errors with high risk content
                if response.status_code == 400:
                    # Reduce max retries to 1/10 of original
                    reduced_max_retries = max(1, max_retries // 20)
                    if times >= reduced_max_retries:
                        print(f"Reached reduced retry limit ({reduced_max_retries}) for high risk content. Giving up.")
                        return {"type": "error", "data": ["Error: Request contains high risk content that cannot be processed."]}
                    print(f"High risk content detected. Using reduced retry limit: {reduced_max_retries}")
                # For other errors
                else:
                    print(f"API request failed with status {response.status_code}: {response.text}")
                
                # Switch to a random API key after several failures
                if times % 3 == 2:  # Every 3rd attempt
                    current_api_key = random.choice(api_keys)
                    headers["Authorization"] = "Bearer " + str(current_api_key)
                    print(f"Switched to random API key: {current_api_key}")
                
            except requests.exceptions.Timeout:
                print(f"Request timed out. Retrying {times+1}/{max_retries}...")
                # Try a different random API key on timeout
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                print(f"Switched to random API key after timeout: {current_api_key}")
            except requests.exceptions.ConnectionError:
                print(f"Connection error. Retrying {times+1}/{max_retries}...")
                # Try a different random API key on connection error
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                print(f"Switched to random API key after connection error: {current_api_key}")
            except Exception as e:
                print(f"Request error: {e}")
            
            # Simple backoff with some randomness
            sleep_time = 1 + random.random()
            print(f"Retrying in {sleep_time:.2f} seconds... in _make_aihubmix_api_request_tool")
            time.sleep(sleep_time)
            times += 1
        
        return {"type": "error", "data": ["Error: Failed to get response after multiple retries."]}

    def solve_problem(self, problem: str, search_tool=None, max_search_calls=None) -> tuple:
        """
        Solve a problem using the policy model with reasoning capabilities.
        
        Args:
            problem: The problem statement to solve
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            
        Returns:
            tuple: (final_answer, message_history) where:
                - final_answer is the extracted answer from the model's response
                - message_history is the complete conversation history
        """
        # Use instance max_search_calls if not explicitly provided
        if max_search_calls is None:
            max_search_calls = self.max_search_calls
            
        # Create messages with system instruction and user problem
        messages = [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": problem
            }
        ]
        
        # Track search calls to prevent infinite loops
        search_call_count = 0
        max_iterations = max_search_calls + 5  # Overall maximum iterations as a safety measure
        current_iteration = 0
        
        # Store search results
        search_results = []
        
        # Main interaction loop
        while True:
            current_iteration += 1
            if current_iteration > max_iterations:
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return "Error: Maximum iterations reached without finding an answer.", messages, search_results
                
            # Make API request using the appropriate method based on use_aihubmix
            if self.use_aihubmix:
                resp = self._make_aihubmix_api_request_tool(
                    messages, 
                    aihubmix_api_keys=self.aihubmix_api_keys,
                    aihubmix_api_url=self.aihubmix_api_url,
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            else:
                resp = self._make_custom_api_request_tool(
                    messages, 
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            
            # Check for errors
            if resp['type'] == 'error':
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return resp['data'][0], messages, search_results
            
            # Process the response
            if resp['type'] == 'tool':
                # Handle tool calls
                tool_list = []
                for tool in resp['data']:
                    func_name = tool['function']['name']
                    tool_call_id = tool['id']
                    try:
                        func_args = json.loads(tool['function']['arguments'])
                    except Exception as e:
                        func_args = tool['function']['arguments']
                    tool_list.append({'name': func_name, 'args': func_args, 'tool_call_id': tool_call_id})
                
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Process each tool call
                for tool in tool_list:
                    # Handle deep_websearch tool calls using search_tool if available
                    if tool['name'] == 'deep_websearch' and search_tool is not None:
                        # Check if we've reached the maximum search calls
                        if search_call_count >= max_search_calls:
                            # Add a message indicating the search limit has been reached
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": f"You have reached the maximum number of search attempts ({max_search_calls}). Please provide your final answer based on the information you have collected so far."
                            })
                            continue
                            
                        search_call_count += 1
                        print(f"Search call {search_call_count}/{max_search_calls}")
                        
                        search_query = tool['args'].get('search_query', '')
                        search_intent = tool['args'].get('search_intent', '')
                        
                        # Skip search if query is empty
                        if not search_query:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": "Error: Empty search query. Please provide a valid search query."
                            })
                            continue
                        
                        # Use search_tool to get search results
                        result = search_tool.search_intent_sync(search_query, search_intent)
                        
                        # Store search result details
                        search_results.append(result)
                        
                        # Add tool result to conversation (only the extracted info)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": result["extracted_info"]
                        })
                    else:
                        # For other tools or if search_tool is not available, use a placeholder
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": f"Tool {tool['name']} was called with arguments {tool['args']}"
                        })
            else:
                # Handle normal text response
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Check for stopping condition: finish_reason is 'stop' and response contains boxed answer
                content = call_messages.get('content', '')
                finish_reason = ''
                
                # Extract finish_reason from the response
                raw_response = resp.get('raw_response', {})
                for choice in raw_response.get('choices', []):
                    finish_reason = choice.get('finish_reason', '')
                    break
                
                # Check if response contains boxed answer
                has_boxed_answer = '\\boxed{' in content
                
                if finish_reason == 'stop' and has_boxed_answer:
                    # Extract the final answer from boxed content
                    import re
                    boxed_pattern = r'\\boxed\{(.*?)\}'
                    matches = re.findall(boxed_pattern, content)
                    
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    if matches:
                        return matches[0], messages, search_results
                    else:
                        return content, messages, search_results
                
                # If we've reached max search calls and still no boxed answer,
                # force a final response after a few more iterations
                if search_call_count >= max_search_calls and current_iteration > max_search_calls + 5:
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    # Extract whatever answer is available, boxed or not
                    if '\\boxed{' in content:
                        import re
                        boxed_pattern = r'\\boxed\{(.*?)\}'
                        matches = re.findall(boxed_pattern, content)
                        if matches:
                            return matches[0], messages, search_results
                    
                    # If no boxed answer, return the entire content
                    return content, messages, search_results

    def solve_problem_budget_forcing(self, messages: List[Dict], search_tool=None, max_search_calls=None) -> tuple:
        """
        Solve a problem with budget forcing.
        """
        if max_search_calls is None:
            max_search_calls = self.max_search_calls

        budget_forcing_prompt_wait = '''I will give you up to a maximum of {max_search_calls} additional chances to use the 'deep_websearch' tool to solve the problem. Trying other solution paths or search strategies is encouraged.'''
        messages.append({
            "role": "user",
            "content": budget_forcing_prompt_wait.format(max_search_calls=max_search_calls)
        })

        # Track search calls to prevent infinite loops
        search_call_count = 0
        max_iterations = max_search_calls + 5  # Overall maximum iterations as a safety measure
        current_iteration = 0
        
        # Store search results
        search_results = []
        
        # Main interaction loop
        while True:
            current_iteration += 1
            if current_iteration > max_iterations:
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return "Error: Maximum iterations reached without finding an answer.", messages, search_results
                
            # Make API request using the appropriate method based on use_aihubmix
            if self.use_aihubmix:
                resp = self._make_aihubmix_api_request_tool(
                    messages, 
                    aihubmix_api_keys=self.aihubmix_api_keys,
                    aihubmix_api_url=self.aihubmix_api_url,
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            else:
                resp = self._make_custom_api_request_tool(
                    messages, 
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            
            # Check for errors
            if resp['type'] == 'error':
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return resp['data'][0], messages, search_results
            
            # Process the response
            if resp['type'] == 'tool':
                # Handle tool calls
                tool_list = []
                for tool in resp['data']:
                    func_name = tool['function']['name']
                    tool_call_id = tool['id']
                    try:
                        func_args = json.loads(tool['function']['arguments'])
                    except Exception as e:
                        func_args = tool['function']['arguments']
                    tool_list.append({'name': func_name, 'args': func_args, 'tool_call_id': tool_call_id})
                
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Process each tool call
                for tool in tool_list:
                    # Handle deep_websearch tool calls using search_tool if available
                    if tool['name'] == 'deep_websearch' and search_tool is not None:
                        # Check if we've reached the maximum search calls
                        if search_call_count >= max_search_calls:
                            # Add a message indicating the search limit has been reached
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": f"You have reached the maximum number of search attempts ({max_search_calls}). Please provide your final answer based on the information you have collected so far."
                            })
                            continue
                            
                        search_call_count += 1
                        print(f"Search call {search_call_count}/{max_search_calls}")
                        
                        search_query = tool['args'].get('search_query', '')
                        search_intent = tool['args'].get('search_intent', '')
                        
                        # Skip search if query is empty
                        if not search_query:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": "Error: Empty search query. Please provide a valid search query."
                            })
                            continue
                        
                        # Use search_tool to get search results
                        result = search_tool.search_intent_sync(search_query, search_intent)
                        
                        # Store search result details
                        search_results.append(result)
                        
                        # Add tool result to conversation (only the extracted info)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": result["extracted_info"]
                        })
                    else:
                        # For other tools or if search_tool is not available, use a placeholder
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": f"Tool {tool['name']} was called with arguments {tool['args']}"
                        })
            else:
                # Handle normal text response
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Check for stopping condition: finish_reason is 'stop' and response contains boxed answer
                content = call_messages.get('content', '')
                finish_reason = ''
                
                # Extract finish_reason from the response
                raw_response = resp.get('raw_response', {})
                for choice in raw_response.get('choices', []):
                    finish_reason = choice.get('finish_reason', '')
                    break
                
                # Check if response contains boxed answer
                has_boxed_answer = '\\boxed{' in content
                
                if finish_reason == 'stop' and has_boxed_answer:
                    # Extract the final answer from boxed content
                    import re
                    boxed_pattern = r'\\boxed\{(.*?)\}'
                    matches = re.findall(boxed_pattern, content)
                    
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    if matches:
                        return matches[0], messages, search_results
                    else:
                        return content, messages, search_results
                
                # If we've reached max search calls and still no boxed answer,
                # force a final response after a few more iterations
                if search_call_count >= max_search_calls and current_iteration > max_search_calls + 5:
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    # Extract whatever answer is available, boxed or not
                    if '\\boxed{' in content:
                        import re
                        boxed_pattern = r'\\boxed\{(.*?)\}'
                        matches = re.findall(boxed_pattern, content)
                        if matches:
                            return matches[0], messages, search_results
                    
                    # If no boxed answer, return the entire content
                    return content, messages, search_results

    def verify_problem(self, problem: str, pred_answer:str, search_tool=None, max_search_calls=None) -> tuple:
        """
        Solve a problem using the policy model with reasoning capabilities.
        
        Args:
            problem: The problem statement to solve
            pred_answer: The predicted answer to the problem
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            
        Returns:
            tuple: (final_answer, message_history) where:
                - final_answer is the extracted answer from the model's response
                - message_history is the complete conversation history
        """
        # Use instance max_search_calls if not explicitly provided
        if max_search_calls is None:
            max_search_calls = self.max_search_calls
            
        # Create messages with system instruction and user problem
        messages = [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": f"[Question Start]: {problem}[Question End]\n[Predicted Answer Start]: {pred_answer}[Predicted Answer End]"
            }
        ]
        
        # Track search calls to prevent infinite loops
        search_call_count = 0
        max_iterations = max_search_calls + 5  # Overall maximum iterations as a safety measure
        current_iteration = 0
        
        # Store search results
        search_results = []
        
        # Main interaction loop
        while True:
            current_iteration += 1
            if current_iteration > max_iterations:
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return "Error: Maximum iterations reached without finding an answer.", messages, search_results
                
            # Make API request using the appropriate method based on use_aihubmix
            if self.use_aihubmix:
                resp = self._make_aihubmix_api_request_tool(
                    messages, 
                    aihubmix_api_keys=self.aihubmix_api_keys,
                    aihubmix_api_url=self.aihubmix_api_url,
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            else:
                resp = self._make_custom_api_request_tool(
                    messages, 
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            
            # Check for errors
            if resp['type'] == 'error':
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return resp['data'][0], messages, search_results
            
            # Process the response
            if resp['type'] == 'tool':
                # Handle tool calls
                tool_list = []
                for tool in resp['data']:
                    func_name = tool['function']['name']
                    tool_call_id = tool['id']
                    try:
                        func_args = json.loads(tool['function']['arguments'])
                    except Exception as e:
                        func_args = tool['function']['arguments']
                    tool_list.append({'name': func_name, 'args': func_args, 'tool_call_id': tool_call_id})
                
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Process each tool call
                for tool in tool_list:
                    # Handle deep_websearch tool calls using search_tool if available
                    if tool['name'] == 'deep_websearch' and search_tool is not None:
                        # Check if we've reached the maximum search calls
                        if search_call_count >= max_search_calls:
                            # Add a message indicating the search limit has been reached
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": f"You have reached the maximum number of search attempts ({max_search_calls}). Please provide your final answer based on the information you have collected so far."
                            })
                            continue
                            
                        search_call_count += 1
                        print(f"Search call {search_call_count}/{max_search_calls}")
                        
                        search_query = tool['args'].get('search_query', '')
                        search_intent = tool['args'].get('search_intent', '')
                        
                        # Skip search if query is empty
                        if not search_query:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": "Error: Empty search query. Please provide a valid search query."
                            })
                            continue
                        
                        # Use search_tool to get search results
                        result = search_tool.search_intent_sync(search_query, search_intent)
                        
                        # Store search result details
                        search_results.append(result)
                        
                        # Add tool result to conversation (only the extracted info)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": result["extracted_info"]
                        })
                    else:
                        # For other tools or if search_tool is not available, use a placeholder
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": f"Tool {tool['name']} was called with arguments {tool['args']}"
                        })
            else:
                # Handle normal text response
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Check for stopping condition: finish_reason is 'stop' and response contains boxed answer
                content = call_messages.get('content', '')
                finish_reason = ''
                
                # Extract finish_reason from the response
                raw_response = resp.get('raw_response', {})
                for choice in raw_response.get('choices', []):
                    finish_reason = choice.get('finish_reason', '')
                    break
                
                # Check if response contains boxed answer
                has_boxed_answer = '\\boxed{' in content
                
                if finish_reason == 'stop' and has_boxed_answer:
                    # Extract the final answer from boxed content
                    import re
                    boxed_pattern = r'\\boxed\{(.*?)\}'
                    matches = re.findall(boxed_pattern, content)
                    
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    if matches:
                        return matches[0], messages, search_results
                    else:
                        return content, messages, search_results
                
                # If we've reached max search calls and still no boxed answer,
                # force a final response after a few more iterations
                if search_call_count >= max_search_calls and current_iteration > max_search_calls + 5:
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    # Extract whatever answer is available, boxed or not
                    if '\\boxed{' in content:
                        import re
                        boxed_pattern = r'\\boxed\{(.*?)\}'
                        matches = re.findall(boxed_pattern, content)
                        if matches:
                            return matches[0], messages, search_results
                    
                    # If no boxed answer, return the entire content
                    return content, messages, search_results
                
                
    def verify_problem_budget_forcing(self, messages: List[Dict], search_tool=None, max_search_calls=None) -> tuple:
        """
        Verify a problem with budget forcing.
        """
        # Use instance max_search_calls if not explicitly provided
        if max_search_calls is None:
            max_search_calls = self.max_search_calls
            
        ###budget_forcing_prompt = '''I will give you additional up to a maximum of {max_search_calls} times to use the "deep_websearch" tool. Please conduct detailed verification to ensure that the score you provide is accurate.'''
        ### budget_forcing_prompt = '''You will be given additional {max_search_calls} times to use the "deep_websearch" tool. Please conduct detailed verification to ensure that the score you provide is accurate.'''
        
        budget_forcing_prompt_wait = '''I will give you up to a maximum of {max_search_calls} additional chances to use the 'deep_websearch' tool. Please conduct detailed verification to ensure that the score you provide is accurate. Trying other verification paths or search strategies is encouraged.'''
        messages.append({
            "role": "user",
            "content": budget_forcing_prompt_wait.format(max_search_calls=max_search_calls)
        })
        
        # Track search calls to prevent infinite loops
        search_call_count = 0
        max_iterations = max_search_calls + 5  # Overall maximum iterations as a safety measure
        current_iteration = 0
        
        # Store search results
        search_results = []
        
        # Main interaction loop
        while True:
            current_iteration += 1
            if current_iteration > max_iterations:
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return "Error: Maximum iterations reached without finding an answer.", messages, search_results
                
            # Make API request using the appropriate method based on use_aihubmix
            if self.use_aihubmix:
                resp = self._make_aihubmix_api_request_tool(
                    messages, 
                    aihubmix_api_keys=self.aihubmix_api_keys,
                    aihubmix_api_url=self.aihubmix_api_url,
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            else:
                resp = self._make_custom_api_request_tool(
                    messages, 
                    tools=self.tools, 
                    tool_choice=self.tool_choice
                )
            
            # Check for errors
            if resp['type'] == 'error':
                # Print token usage before returning
                if self.use_aihubmix:
                    print("\n===== FINAL TOKEN USAGE =====")
                    print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                    print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                    print(f"Total tokens: {self.token_counters['total_tokens']}")
                    print("============================")
                return resp['data'][0], messages, search_results
            
            # Process the response
            if resp['type'] == 'tool':
                # Handle tool calls
                tool_list = []
                for tool in resp['data']:
                    func_name = tool['function']['name']
                    tool_call_id = tool['id']
                    try:
                        func_args = json.loads(tool['function']['arguments'])
                    except Exception as e:
                        func_args = tool['function']['arguments']
                    tool_list.append({'name': func_name, 'args': func_args, 'tool_call_id': tool_call_id})
                
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Process each tool call
                for tool in tool_list:
                    # Handle deep_websearch tool calls using search_tool if available
                    if tool['name'] == 'deep_websearch' and search_tool is not None:
                        # Check if we've reached the maximum search calls
                        if search_call_count >= max_search_calls:
                            # Add a message indicating the search limit has been reached
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": f"You have reached the maximum number of search attempts ({max_search_calls}). Please provide your final answer based on the information you have collected so far."
                            })
                            continue
                            
                        search_call_count += 1
                        print(f"Search call {search_call_count}/{max_search_calls}")
                        
                        search_query = tool['args'].get('search_query', '')
                        search_intent = tool['args'].get('search_intent', '')
                        
                        # Skip search if query is empty
                        if not search_query:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool['tool_call_id'],
                                "content": "Error: Empty search query. Please provide a valid search query."
                            })
                            continue
                        
                        # Use search_tool to get search results
                        result = search_tool.search_intent_sync(search_query, search_intent)
                        
                        # Store search result details
                        search_results.append(result)
                        
                        # Add tool result to conversation (only the extracted info)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": result["extracted_info"]
                        })
                    else:
                        # For other tools or if search_tool is not available, use a placeholder
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool['tool_call_id'],
                            "content": f"Tool {tool['name']} was called with arguments {tool['args']}"
                        })
            else:
                # Handle normal text response
                call_messages = resp['call_messages']
                
                # Add assistant's message to conversation
                messages.append(call_messages)
                
                # Check for stopping condition: finish_reason is 'stop' and response contains boxed answer
                content = call_messages.get('content', '')
                finish_reason = ''
                
                # Extract finish_reason from the response
                raw_response = resp.get('raw_response', {})
                for choice in raw_response.get('choices', []):
                    finish_reason = choice.get('finish_reason', '')
                    break
                
                # Check if response contains boxed answer
                has_boxed_answer = '\\boxed{' in content
                
                if finish_reason == 'stop' and has_boxed_answer:
                    # Extract the final answer from boxed content
                    import re
                    boxed_pattern = r'\\boxed\{(.*?)\}'
                    matches = re.findall(boxed_pattern, content)
                    
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    if matches:
                        return matches[0], messages, search_results
                    else:
                        return content, messages, search_results
                
                # If we've reached max search calls and still no boxed answer,
                # force a final response after a few more iterations
                if search_call_count >= max_search_calls and current_iteration > max_search_calls + 5:
                    # Print token usage before returning
                    if self.use_aihubmix:
                        print("\n===== FINAL TOKEN USAGE =====")
                        print(f"Prompt tokens: {self.token_counters['prompt_tokens']}")
                        print(f"Completion tokens: {self.token_counters['completion_tokens']}")
                        print(f"Total tokens: {self.token_counters['total_tokens']}")
                        print("============================")
                    
                    # Extract whatever answer is available, boxed or not
                    if '\\boxed{' in content:
                        import re
                        boxed_pattern = r'\\boxed\{(.*?)\}'
                        matches = re.findall(boxed_pattern, content)
                        if matches:
                            return matches[0], messages, search_results
                    
                    # If no boxed answer, return the entire content
                    return content, messages, search_results

    async def solve_problems_parallel(self, problems: List[str], search_tool=None, max_search_calls=None, concurrent_limit=10) -> List[tuple]:
        """
        Solve multiple problems in parallel using the policy model.
        
        Args:
            problems: List of problem statements to solve
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            
        Returns:
            List of tuples: [(final_answer, message_history), ...] for each problem
        """
        import asyncio
        from tqdm.asyncio import tqdm_asyncio
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def solve_single_problem(problem):
            """Wrapper to solve a single problem with semaphore"""
            async with semaphore:
                # Create a thread to run the synchronous solve_problem method
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.solve_problem(problem, search_tool, max_search_calls)
                )
        
        # Create tasks for all problems
        tasks = [solve_single_problem(problem) for problem in problems]
        
        # Execute all tasks with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Solving problems")
        
        return results
    
    def solve_problems_parallel_sync(self, problems: List[str], search_tool=None, max_search_calls=None, concurrent_limit=10) -> List[tuple]:
        """
        Synchronous wrapper for solve_problems_parallel.
        
        Args:
            problems: List of problem statements to solve
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            
        Returns:
            List of tuples: [(final_answer, message_history, search_results), ...] for each problem
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.solve_problems_parallel(problems, search_tool, max_search_calls, concurrent_limit)
            )
            loop.close()
            return results
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            return [(f"Error: {str(e)}", [], []) for _ in problems]

    async def process_batch(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                           concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Process a batch of problems with advanced features like periodic saving and cancellation handling.
        
        Args:
            problems: List of problem dictionaries, each containing at least a 'question' key
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with answers
        """
        import asyncio
        import os
        import json
        import time
        import signal
        from tqdm.asyncio import tqdm_asyncio
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique filename for results
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        result_file = os.path.join(output_dir, f"policy_results_{timestamp}.json")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        # Create a lock for saving results
        lock = asyncio.Lock()
        
        # List to store processed problems
        processed_problems = []
        
        # Set up cancellation handling
        shutdown_event = asyncio.Event()
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        
        def sigint_handler(sig, frame):
            print("\nReceived SIGINT. Attempting graceful shutdown...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, sigint_handler)
        
        async def solve_single_problem(problem_dict):
            """Process a single problem with error handling"""
            if shutdown_event.is_set():
                return None
                
            try:
                async with semaphore:
                    if shutdown_event.is_set():
                        return None
                        
                    # Extract the question from the problem dictionary
                    question = problem_dict.get('question', problem_dict.get('Question', ''))
                    
                    # Create a thread to run the synchronous solve_problem method
                    loop = asyncio.get_running_loop()
                    answer, messages, search_results = await loop.run_in_executor(
                        None, 
                        lambda: self.solve_problem(question, search_tool, max_search_calls)
                    )
                    
                    # Update the problem dictionary with results
                    result_dict = problem_dict.copy()
                    result_dict['pred_answer'] = answer
                    result_dict['messages'] = messages
                    result_dict['web_search'] = search_results
                    
                    # Save periodically
                    async with lock:
                        processed_problems.append(result_dict)
                        if len(processed_problems) % save_interval == 0:
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
                    
                    return result_dict
            except Exception as e:
                print(f"Error processing problem: {str(e)}")
                # Create a minimal result with error information
                result_dict = problem_dict.copy()
                result_dict['pred_answer'] = f"Error: {str(e)}"
                result_dict['error'] = str(e)
                return result_dict
        
        try:
            # Create tasks for all problems
            tasks = [solve_single_problem(problem) for problem in problems]
            
            # Execute all tasks with progress bar
            results = await tqdm_asyncio.gather(*tasks, desc="Processing problems")
            
            # Filter out None results (from cancelled tasks)
            results = [r for r in results if r is not None]
            
        except asyncio.CancelledError:
            print("Tasks were cancelled - saving partial results")
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Save final results
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
        
        return processed_problems
    
    def process_batch_sync(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                          concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Synchronous wrapper for process_batch.
        
        Args:
            problems: List of problem dictionaries, each containing at least a 'question' key
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with answers
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.process_batch(problems, search_tool, max_search_calls, concurrent_limit, output_dir, save_interval)
            )
            loop.close()
            
            # Print final token usage after batch processing
            if self.use_aihubmix:
                print("\n===== FINAL BATCH TOKEN USAGE =====")
                print(f"Total prompt tokens: {self.token_counters['prompt_tokens']}")
                print(f"Total completion tokens: {self.token_counters['completion_tokens']}")
                print(f"Total tokens: {self.token_counters['total_tokens']}")
                print("=================================")
                
                # Save token usage and cost information to file
                self.save_statistics(output_dir, search_tool)
                
            return results
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return []

    async def verify_problems_parallel(self, problems: List[str], pred_answers: List[str], search_tool=None, max_search_calls=None, concurrent_limit=10) -> List[tuple]:
        """
        Verify multiple problems in parallel using the policy model.
        
        Args:
            problems: List of problem statements to verify
            pred_answers: List of predicted answers to verify
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            
        Returns:
            List of tuples: [(confidence_score, message_history, search_results), ...] for each problem
        """
        import asyncio
        from tqdm.asyncio import tqdm_asyncio
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def verify_single_problem(problem, pred_answer):
            """Wrapper to verify a single problem with semaphore"""
            async with semaphore:
                # Create a thread to run the synchronous verify_problem method
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.verify_problem(problem, pred_answer, search_tool, max_search_calls)
                )
        
        # Create tasks for all problems
        tasks = [verify_single_problem(problem, pred_answer) 
                for problem, pred_answer in zip(problems, pred_answers)]
        
        # Execute all tasks with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Verifying problems")
        
        return results
    
    def verify_problems_parallel_sync(self, problems: List[str], pred_answers: List[str], search_tool=None, max_search_calls=None, concurrent_limit=10) -> List[tuple]:
        """
        Synchronous wrapper for verify_problems_parallel.
        
        Args:
            problems: List of problem statements to verify
            pred_answers: List of predicted answers to verify
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            
        Returns:
            List of tuples: [(confidence_score, message_history, search_results), ...] for each problem
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.verify_problems_parallel(problems, pred_answers, search_tool, max_search_calls, concurrent_limit)
            )
            loop.close()
            return results
        except Exception as e:
            print(f"Error in parallel verification: {str(e)}")
            return [(f"Error: {str(e)}", [], []) for _ in problems]

    async def verify_batch(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                          concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Process a batch of problems with verification, including advanced features like periodic saving and cancellation handling.
        
        Args:
            problems: List of problem dictionaries, each containing at least 'question' and 'pred_answer' keys
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with verification scores
        """
        import asyncio
        import os
        import json
        import time
        import signal
        from tqdm.asyncio import tqdm_asyncio
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique filename for results
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        result_file = os.path.join(output_dir, f"verification_results_{timestamp}.json")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        # Create a lock for saving results
        lock = asyncio.Lock()
        
        # List to store processed problems
        processed_problems = []
        
        # Set up cancellation handling
        shutdown_event = asyncio.Event()
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        
        def sigint_handler(sig, frame):
            print("\nReceived SIGINT. Attempting graceful shutdown...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, sigint_handler)
        
        async def verify_single_problem(problem_dict):
            """Verify a single problem with error handling"""
            if shutdown_event.is_set():
                return None
                
            try:
                async with semaphore:
                    if shutdown_event.is_set():
                        return None
                        
                    # Extract the question and predicted answer from the problem dictionary
                    question = problem_dict.get('question', problem_dict.get('Question', ''))
                    pred_answer = problem_dict.get('pred_answer', problem_dict.get('answer', ''))
                    
                    # Create a thread to run the synchronous verify_problem method
                    loop = asyncio.get_running_loop()
                    confidence_score, messages, search_results = await loop.run_in_executor(
                        None, 
                        lambda: self.verify_problem(question, pred_answer, search_tool, max_search_calls)
                    )
                    
                    # Update the problem dictionary with results
                    result_dict = problem_dict.copy()
                    result_dict['confidence_score'] = confidence_score
                    result_dict['messages'] = messages
                    result_dict['verification_searches'] = search_results
                    
                    # Save periodically
                    async with lock:
                        processed_problems.append(result_dict)
                        if len(processed_problems) % save_interval == 0:
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
                    
                    return result_dict
            except Exception as e:
                print(f"Error verifying problem: {str(e)}")
                # Create a minimal result with error information
                result_dict = problem_dict.copy()
                result_dict['confidence_score'] = f"Error: {str(e)}"
                result_dict['error'] = str(e)
                return result_dict
        
        try:
            # Create tasks for all problems
            tasks = [verify_single_problem(problem) for problem in problems]
            
            # Execute all tasks with progress bar
            results = await tqdm_asyncio.gather(*tasks, desc="Verifying problems")
            
            # Filter out None results (from cancelled tasks)
            results = [r for r in results if r is not None]
            
        except asyncio.CancelledError:
            print("Tasks were cancelled - saving partial results")
        except Exception as e:
            print(f"Error in batch verification: {str(e)}")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Save final results
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
        
        return processed_problems
    
    def verify_batch_sync(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                         concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Synchronous wrapper for verify_batch.
        
        Args:
            problems: List of problem dictionaries, each containing at least 'question' and 'pred_answer' keys
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with verification scores
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.verify_batch(problems, search_tool, max_search_calls, concurrent_limit, output_dir, save_interval)
            )
            loop.close()
            
            # Print final token usage after batch verification
            if self.use_aihubmix:
                print("\n===== FINAL BATCH TOKEN USAGE =====")
                print(f"Total prompt tokens: {self.token_counters['prompt_tokens']}")
                print(f"Total completion tokens: {self.token_counters['completion_tokens']}")
                print(f"Total tokens: {self.token_counters['total_tokens']}")
                print("=================================")
                
                # Save token usage and cost information to file
                self.save_statistics(output_dir, search_tool)
                
            return results
        except Exception as e:
            print(f"Error in batch verification: {str(e)}")
            return []

    def reset_token_counters(self):
        """Reset all token counters to zero."""
        self.token_counters = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

    def save_statistics(self, output_dir, search_tool=None):
        """
        Save token usage, API counters, and cost information to a file in the output directory.
        
        Args:
            output_dir: Directory to save the statistics file
            search_tool: Optional WebSearchTool instance to include its API counters
        """
        import os
        import json
        import time
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        stats_file = os.path.join(output_dir, f"statistics_{timestamp}.json")
        
        # Prepare statistics dictionary
        statistics = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "aux_model_name": self.aux_model_name,
            "token_usage": self.token_counters.copy(),
            "api_counters": {k: v for k, v in self.api_counters.items()}
        }
        
        # Add search tool API counters if available
        if search_tool is not None and hasattr(search_tool, 'api_counters'):
            statistics["search_tool_api_counters"] = {k: v for k, v in search_tool.api_counters.items()}
        
        # Calculate and add cost information if available
        cost_info = self.calculate_cost()
        if cost_info:
            statistics["cost_information"] = cost_info
        
        # Save to file
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            print(f"Statistics saved to {stats_file}")
        except Exception as e:
            print(f"Error saving statistics to file: {e}")

    async def verify_problems_budget_forcing_parallel(self, problems_messages: List[List[Dict]], search_tool=None, max_search_calls=None, concurrent_limit=10) -> List[tuple]:
        """
        Verify multiple problems in parallel using the budget forcing approach.
        
        Args:
            problems_messages: List of message histories for each problem
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            
        Returns:
            List of tuples: [(confidence_score, message_history, search_results), ...] for each problem
        """
        import asyncio
        from tqdm.asyncio import tqdm_asyncio
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def verify_single_problem(messages):
            """Wrapper to verify a single problem with semaphore"""
            async with semaphore:
                # Create a thread to run the synchronous verify_problem_budget_forcing method
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.verify_problem_budget_forcing(messages, search_tool, max_search_calls)
                )
        
        # Create tasks for all problems
        tasks = [verify_single_problem(messages) for messages in problems_messages]
        
        # Execute all tasks with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Verifying problems with budget forcing")
        
        return results
    
    def verify_problems_budget_forcing_parallel_sync(self, problems_messages: List[List[Dict]], search_tool=None, max_search_calls=None, concurrent_limit=10) -> List[tuple]:
        """
        Synchronous wrapper for verify_problems_budget_forcing_parallel.
        
        Args:
            problems_messages: List of message histories for each problem
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            
        Returns:
            List of tuples: [(confidence_score, message_history, search_results), ...] for each problem
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.verify_problems_budget_forcing_parallel(problems_messages, search_tool, max_search_calls, concurrent_limit)
            )
            loop.close()
            return results
        except Exception as e:
            print(f"Error in parallel budget forcing verification: {str(e)}")
            return [(f"Error: {str(e)}", [], []) for _ in problems_messages]

    async def verify_batch_budget_forcing(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                          concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Process a batch of problems with budget forcing verification, including advanced features like periodic saving and cancellation handling.
        
        Args:
            problems: List of problem dictionaries, each containing at least 'messages' key
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with verification scores
        """
        import asyncio
        import os
        import json
        import time
        import signal
        from tqdm.asyncio import tqdm_asyncio
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique filename for results
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        result_file = os.path.join(output_dir, f"budget_forcing_verification_results_{timestamp}.json")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        # Create a lock for saving results
        lock = asyncio.Lock()
        
        # List to store processed problems
        processed_problems = []
        
        # Set up cancellation handling
        shutdown_event = asyncio.Event()
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        
        def sigint_handler(sig, frame):
            print("\nReceived SIGINT. Attempting graceful shutdown...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, sigint_handler)
        
        async def verify_single_problem(problem_dict):
            """Verify a single problem with error handling"""
            if shutdown_event.is_set():
                return None
                
            try:
                async with semaphore:
                    if shutdown_event.is_set():
                        return None
                        
                    # Extract the messages from the problem dictionary
                    messages = problem_dict.get('messages', [])
                    
                    # Create a thread to run the synchronous verify_problem_budget_forcing method
                    loop = asyncio.get_running_loop()
                    confidence_score, updated_messages, search_results = await loop.run_in_executor(
                        None, 
                        lambda: self.verify_problem_budget_forcing(messages, search_tool, max_search_calls)
                    )
                    
                    # Update the problem dictionary with results
                    result_dict = problem_dict.copy()
                    result_dict['confidence_score'] = confidence_score
                    result_dict['messages'] = updated_messages
                    result_dict['verification_budget_searches'] = search_results
                    
                    # Save periodically
                    async with lock:
                        processed_problems.append(result_dict)
                        if len(processed_problems) % save_interval == 0:
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
                    
                    return result_dict
            except Exception as e:
                print(f"Error verifying problem: {str(e)}")
                # Create a minimal result with error information
                result_dict = problem_dict.copy()
                result_dict['confidence_score'] = f"Error: {str(e)}"
                result_dict['error'] = str(e)
                return result_dict
        
        try:
            # Create tasks for all problems
            tasks = [verify_single_problem(problem) for problem in problems]
            
            # Execute all tasks with progress bar
            results = await tqdm_asyncio.gather(*tasks, desc="Verifying problems with budget forcing")
            
            # Filter out None results (from cancelled tasks)
            results = [r for r in results if r is not None]
            
        except asyncio.CancelledError:
            print("Tasks were cancelled - saving partial results")
        except Exception as e:
            print(f"Error in batch budget forcing verification: {str(e)}")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Save final results
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
            
            # Save token usage and cost information
            if self.use_aihubmix:
                print("\n===== FINAL BATCH TOKEN USAGE =====")
                print(f"Total prompt tokens: {self.token_counters['prompt_tokens']}")
                print(f"Total completion tokens: {self.token_counters['completion_tokens']}")
                print(f"Total tokens: {self.token_counters['total_tokens']}")
                print(f"Estimated cost: ${self.calculate_cost()}")
                print("============================")
                
                # Save statistics to file
                self.save_statistics(output_dir, search_tool)
        
        return processed_problems
    
    def verify_batch_budget_forcing_sync(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                         concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Synchronous wrapper for verify_batch_budget_forcing.
        
        Args:
            problems: List of problem dictionaries, each containing at least 'messages' key
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with verification scores
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.verify_batch_budget_forcing(problems, search_tool, max_search_calls, concurrent_limit, output_dir, save_interval)
            )
            loop.close()
            
            # Print final token usage after batch verification
            if self.use_aihubmix:
                print("\n===== FINAL BATCH TOKEN USAGE =====")
                print(f"Total prompt tokens: {self.token_counters['prompt_tokens']}")
                print(f"Total completion tokens: {self.token_counters['completion_tokens']}")
                print(f"Total tokens: {self.token_counters['total_tokens']}")
                print(f"Estimated cost: ${self.calculate_cost()}")
                print("============================")
            
            return results
        except Exception as e:
            print(f"Error in batch budget forcing verification: {str(e)}")
            return []

    async def solve_batch_budget_forcing(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                          concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Process a batch of problems with budget forcing, including advanced features like periodic saving and cancellation handling.
        
        Args:
            problems: List of problem dictionaries, each containing at least 'messages' key
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with answers
        """
        import asyncio
        import os
        import json
        import time
        import signal
        from tqdm.asyncio import tqdm_asyncio
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique filename for results
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        result_file = os.path.join(output_dir, f"budget_forcing_solving_results_{timestamp}.json")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        # Create a lock for saving results
        lock = asyncio.Lock()
        
        # List to store processed problems
        processed_problems = []
        
        # Set up cancellation handling
        shutdown_event = asyncio.Event()
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        
        def sigint_handler(sig, frame):
            print("\nReceived SIGINT. Attempting graceful shutdown...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, sigint_handler)
        
        async def solve_single_problem(problem_dict):
            """Solve a single problem with error handling"""
            if shutdown_event.is_set():
                return None
                
            try:
                async with semaphore:
                    if shutdown_event.is_set():
                        return None
                        
                    # Extract the messages from the problem dictionary
                    messages = problem_dict.get('messages', [])
                    
                    # Create a thread to run the synchronous solve_problem_budget_forcing method
                    loop = asyncio.get_running_loop()
                    answer, updated_messages, search_results = await loop.run_in_executor(
                        None, 
                        lambda: self.solve_problem_budget_forcing(messages, search_tool, max_search_calls)
                    )
                    
                    # Update the problem dictionary with results
                    result_dict = problem_dict.copy()
                    result_dict['pred_answer'] = answer
                    result_dict['messages'] = updated_messages
                    result_dict['solving_budget_searches'] = search_results
                    
                    # Save periodically
                    async with lock:
                        processed_problems.append(result_dict)
                        if len(processed_problems) % save_interval == 0:
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(processed_problems, f, ensure_ascii=False, indent=2)
                    
                    return result_dict
            except Exception as e:
                print(f"Error solving problem: {str(e)}")
                # Create a minimal result with error information
                result_dict = problem_dict.copy()
                result_dict['pred_answer'] = f"Error: {str(e)}"
                result_dict['error'] = str(e)
                return result_dict
        
        try:
            # Create tasks for all problems
            tasks = [solve_single_problem(problem) for problem in problems]
            
            # Execute all tasks with progress bar
            results = await tqdm_asyncio.gather(*tasks, desc="Solving problems with budget forcing")
            
            # Filter out None results (from cancelled tasks)
            results = [r for r in results if r is not None]
            
            # Save final results
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            # Restore original SIGINT handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Print token usage after batch solving
            if self.use_aihubmix:
                print("\n===== FINAL BATCH TOKEN USAGE =====")
                print(f"Total prompt tokens: {self.token_counters['prompt_tokens']}")
                print(f"Total completion tokens: {self.token_counters['completion_tokens']}")
                print(f"Total tokens: {self.token_counters['total_tokens']}")
                print(f"Estimated cost: ${self.calculate_cost()}")
                print("============================")
            
            return results
            
        except Exception as e:
            # Restore original SIGINT handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            print(f"Batch solving error: {str(e)}")
            
            # Save whatever results we have
            if processed_problems:
                error_file = os.path.join(output_dir, f"budget_forcing_solving_results_error_{timestamp}.json")
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_problems, f, ensure_ascii=False, indent=2)
                    
            return processed_problems

    def solve_batch_budget_forcing_sync(self, problems: List[Dict], search_tool=None, max_search_calls=None, 
                         concurrent_limit=10, output_dir="./outputs", save_interval=5):
        """
        Synchronous wrapper for solve_batch_budget_forcing.
        
        Args:
            problems: List of problem dictionaries, each containing at least 'messages' key
            search_tool: Optional WebSearchTool instance for handling search requests
            max_search_calls: Optional override for maximum number of search tool calls allowed
            concurrent_limit: Maximum number of concurrent problems to process
            output_dir: Directory to save results
            save_interval: How often to save intermediate results (every N problems)
            
        Returns:
            List of processed problem dictionaries with answers
        """
        import asyncio
        
        # Create and run an event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.solve_batch_budget_forcing(problems, search_tool, max_search_calls, concurrent_limit, output_dir, save_interval)
            )
            loop.close()
            
            # Print final token usage after batch solving
            if self.use_aihubmix:
                print("\n===== FINAL BATCH TOKEN USAGE =====")
                print(f"Total prompt tokens: {self.token_counters['prompt_tokens']}")
                print(f"Total completion tokens: {self.token_counters['completion_tokens']}")
                print(f"Total tokens: {self.token_counters['total_tokens']}")
                print(f"Estimated cost: ${self.calculate_cost()}")
                print("============================")
            
            return results
        except Exception as e:
            print(f"Error in batch budget forcing solving: {str(e)}")
            return []

class WebSearchTool:
    """Tool for performing web searches and extracting information."""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        aux_api_base_url: str = "http://localhost:8000",
        model_name: str = "QwQ-32B",
        aux_model_name: str = "Qwen2.5-32B-Instruct",
        api_key: str = "empty",
        aux_api_key: str = "empty",
        bing_subscription_key: str = "empty_key",
        bing_endpoint: str = "https://api.bing.microsoft.com/v7.0/search",
        use_bing_pro: bool = False,
        bing_pro_token: str = "1791013312122257441",
        bing_pro_api: str = "bing-search-pro",
        use_google_pro: bool = False,
        google_pro_api_key: str = "81b2d7ef2974da1a63669e7ffa5534a6974ff990",
        use_jina: bool = False,
        jina_api_key: Optional[str] = None,
        keep_links: bool = False,
        top_k: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.8,
        min_p: float = 0.05,
        top_k_sampling: int = 20,
        repetition_penalty: float = 1.05,
        max_tokens: int = 81920,
        concurrent_limit: int = 32,
        model_concurrent_limit: Optional[int] = None,
        bing_concurrent_limit: Optional[int] = None,
        aux_model_concurrent_limit: Optional[int] = None,
        use_custom_api: bool = False,
        custom_api_url: Optional[str] = None,
        use_aihubmix: bool = False,
        aihubmix_api_url: Optional[str] = None,
        aihubmix_api_keys: Optional[str] = None,
        cache_dir: str = './cache',
    ):
        """Initialize the WebSearchTool with configuration parameters."""
        # Store configuration
        self.api_base_url = api_base_url
        self.aux_api_base_url = aux_api_base_url
        self.model_name = model_name
        self.aux_model_name = aux_model_name
        self.api_key = api_key
        self.aux_api_key = aux_api_key
        self.bing_subscription_key = bing_subscription_key
        self.bing_endpoint = bing_endpoint
        self.use_bing_pro = use_bing_pro
        self.bing_pro_token = bing_pro_token
        self.bing_pro_api = bing_pro_api
        self.use_google_pro = use_google_pro
        self.google_pro_api_key = google_pro_api_key
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
        self.keep_links = keep_links
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k_sampling = top_k_sampling
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.use_custom_api = use_custom_api
        self.custom_api_url = custom_api_url
        self.use_aihubmix = use_aihubmix
        self.aihubmix_api_url = aihubmix_api_url
        self.aihubmix_api_keys = aihubmix_api_keys
        
        # Set up concurrency limits
        self.model_concurrent_limit = model_concurrent_limit if model_concurrent_limit is not None else concurrent_limit
        self.bing_concurrent_limit = bing_concurrent_limit if bing_concurrent_limit is not None else concurrent_limit
        self.aux_model_concurrent_limit = aux_model_concurrent_limit if aux_model_concurrent_limit is not None else concurrent_limit
        
        # Initialize API clients
        self.client = None
        self.aux_client = None
        if not use_custom_api:
            init_api_key = api_key.split(',')[0] if ',' in api_key else api_key
            init_aux_api_key = aux_api_key.split(',')[0] if ',' in aux_api_key else aux_api_key
            
            self.client = AsyncOpenAI(
                api_key=init_api_key,
                base_url=api_base_url,
            )
            self.aux_client = AsyncOpenAI(
                api_key=init_aux_api_key,
                base_url=aux_api_base_url,
            )
        
        # API counters
        self.api_counters = Counter({
            'main_model': 0,
            'aux_model': 0,
            'bing_search': 0,
            'bing_search_pro': 0,
            'google_search_pro': 0,
            'page_fetch': 0,
        })
    
    def get_random_key(self, api_key):
        """Get a random key from a comma-separated list of keys."""
        if api_key and ',' in api_key:
            keys = api_key.split(',')
            return random.choice(keys)
        return api_key
    
    async def search(self, query: str, context: str = "") -> str:
        """
        Perform a web search for the given query and return extracted information.
        
        Args:
            query: The search query
            context: Optional context to help guide the search
            
        Returns:
            Extracted information from search results
        """
        # Create semaphores for concurrent operations
        model_semaphore = asyncio.Semaphore(self.model_concurrent_limit)
        bing_semaphore = asyncio.Semaphore(self.bing_concurrent_limit)
        aux_model_semaphore = asyncio.Semaphore(self.aux_model_concurrent_limit)
        
        # Generate search intent
        try:
            _, search_intent = await generate_response(
                prompt=get_search_intent_instruction(context + query),
                model_name=self.aux_model_name,
                client=self.aux_client,
                semaphore=aux_model_semaphore,
                max_tokens=1000,
                api_key=self.aux_api_key,
                use_custom_api=self.use_custom_api,
                custom_api_url=self.custom_api_url,
                use_aihubmix=self.use_aihubmix,
                aihubmix_api_url=self.aihubmix_api_url,
                aihubmix_api_keys=self.aihubmix_api_keys,
                api_counters=self.api_counters,
                args=self,
            )
        except Exception as e:
            print(f"Error generating search intent: {e}")
            search_intent = "Find relevant information about: " + query
        
        # Execute search
        try:
            if self.use_google_pro:
                # Use Google Search Pro API
                results = await google_web_search_async_pro(
                    query, 
                    api_key=self.google_pro_api_key,
                    semaphore=bing_semaphore,
                    api_counters=self.api_counters,
                )
            elif self.use_bing_pro:
                # Use Bing Search Pro API
                results = await bing_web_search_async_pro(
                    query, 
                    token=self.bing_pro_token, 
                    api=self.bing_pro_api,
                    semaphore=bing_semaphore,
                    api_counters=self.api_counters,
                )
            else:
                # Use standard Bing Search API
                results = await bing_web_search_async(
                    query, 
                    self.bing_subscription_key, 
                    self.bing_endpoint,
                    semaphore=bing_semaphore,
                    api_counters=self.api_counters,
                )
        except Exception as e:
            print(f"Error during search query '{query}': {e}")
            return f"Error: Failed to search for '{query}': {str(e)}"
        
        # Extract relevant information - always use extract_relevant_info_pro for Google Pro and Bing Pro
        if self.use_google_pro or self.use_bing_pro:
            relevant_info = extract_relevant_info_pro(results)[:self.top_k]
        else:
            relevant_info = extract_relevant_info(results)[:self.top_k]
        
        # Process documents - fetch content for each URL
        urls_to_fetch = [doc_info['url'] for doc_info in relevant_info]
        
        if urls_to_fetch:
            try:
                contents = await fetch_page_content_async(
                    urls_to_fetch, 
                    use_jina=self.use_jina, 
                    jina_api_key=self.jina_api_key, 
                    keep_links=self.keep_links,
                    api_counters=self.api_counters,
                    max_concurrent=200,
                    show_progress=False,
                )
            except Exception as e:
                print(f"Error fetching URLs: {e}")
                contents = {}
        else:
            contents = {}
        
        # Get web page information for each result
        for doc_info in relevant_info:
            url = doc_info['url']
            raw_content = contents.get(url, "")
                
            if raw_content:
                is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)
            
            # Check if content has error indicators
            has_error = any(indicator.lower() in raw_content.lower() for indicator in error_indicators) or raw_content == "" or raw_content.startswith("Error:")
            
            if has_error:
                # If content has error, use it directly as summary
                doc_info['page_info'] = "Can not fetch the page content."
            else:
                # Use raw content directly as page info
                doc_info['page_info'] = raw_content
        
        # Format search results
        formatted_documents = format_search_results(relevant_info)
        
        # Generate deep web exploration
        try:
            # Create an args object to pass to generate_deep_web_explorer
            class Args:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            args = Args(
                model_name=self.model_name,
                aux_model_name=self.aux_model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                repetition_penalty=self.repetition_penalty,
                top_k_sampling=self.top_k_sampling,
                min_p=self.min_p,
                use_jina=self.use_jina,
                jina_api_key=self.jina_api_key,
                keep_links=self.keep_links,
                use_bing_pro=self.use_bing_pro,
                bing_pro_token=self.bing_pro_token,
                bing_pro_api=self.bing_pro_api,
                use_google_pro=self.use_google_pro,
                google_pro_api_key=self.google_pro_api_key,
                use_custom_api=self.use_custom_api,
                custom_api_url=self.custom_api_url,
                use_aihubmix=self.use_aihubmix,
                aihubmix_api_url=self.aihubmix_api_url,
                aihubmix_api_keys=self.aihubmix_api_keys,
                api_key=self.api_key,
                aux_api_key=self.aux_api_key,
                top_k=self.top_k,
            )
            
            # Create empty caches for the deep web explorer since we're not using caching
            empty_search_cache = {}
            empty_url_cache = {}
            empty_cache_lock = asyncio.Lock()
            
            # Create semaphore for shared use
            semaphore = asyncio.Semaphore(self.model_concurrent_limit)
            
            analysis, _ = await generate_deep_web_explorer(
                client=self.client,
                aux_client=self.aux_client,
                search_query=query,
                search_intent=search_intent,
                document=formatted_documents,
                args=args,
                search_cache=empty_search_cache,
                url_cache=empty_url_cache,
                search_cache_lock=empty_cache_lock,
                url_cache_lock=empty_cache_lock,
                semaphore=semaphore,
                model_semaphore=model_semaphore,
                bing_semaphore=bing_semaphore,
                aux_model_semaphore=aux_model_semaphore,
                api_counters=self.api_counters,
                use_aihubmix=args.use_aihubmix if hasattr(args, 'use_aihubmix') else False,
                aihubmix_api_url=args.aihubmix_api_url if hasattr(args, 'aihubmix_api_url') else None,
                aihubmix_api_keys=args.aihubmix_api_keys if hasattr(args, 'aihubmix_api_keys') else None,
            )
            
            # Extract answer from the analysis
            extracted_info = extract_answer_fn(analysis, mode='summary')
            
            return extracted_info
        except Exception as e:
            print(f"Error in deep web exploration: {e}")
            # Fallback: return the formatted documents directly
            return f"Error in deep exploration: {str(e)}\n\nSearch Results:\n{formatted_documents}"
    
    def search_sync(self, query: str, context: str = "") -> str:
        """
        Synchronous version of the search method.
        
        Args:
            query: The search query
            context: Optional context to help guide the search
            
        Returns:
            Extracted information from search results
        """
        # Create and run an event loop to execute the async search
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.search(query, context))
            loop.close()
            return result
        except Exception as e:
            print(f"Error executing search: {str(e)}")
            return f"Error executing search: {str(e)}"

    async def search_with_intent(self, query: str, intent: str) -> dict:
        """
        Perform a web search for the given query and provided intent, returning extracted information.
        Args:
            query: The search query
            intent: The search intent (already generated)
        Returns:
            Dictionary containing extracted information and search details
        """
        # Create semaphores for concurrent operations
        model_semaphore = asyncio.Semaphore(self.model_concurrent_limit)
        bing_semaphore = asyncio.Semaphore(self.bing_concurrent_limit)
        aux_model_semaphore = asyncio.Semaphore(self.aux_model_concurrent_limit)

        # Use provided intent directly
        search_intent = intent

        # Execute search
        try:
            if self.use_google_pro:
                # Use Google Search Pro API
                results = await google_web_search_async_pro(
                    query, 
                    api_key=self.google_pro_api_key,
                    semaphore=bing_semaphore,
                    api_counters=self.api_counters,
                )
            elif self.use_bing_pro:
                results = await bing_web_search_async_pro(
                    query,
                    token=self.bing_pro_token,
                    api=self.bing_pro_api,
                    semaphore=bing_semaphore,
                    api_counters=self.api_counters,
                )
            else:
                results = await bing_web_search_async(
                    query,
                    self.bing_subscription_key,
                    self.bing_endpoint,
                    semaphore=bing_semaphore,
                    api_counters=self.api_counters,
                )
        except Exception as e:
            print(f"Error during search query '{query}': {e}")
            return {
                "extracted_info": f"Error: Failed to search for '{query}': {str(e)}",
                "search_query": query,
                "search_intent": search_intent,
                "formatted_documents": "",
                "analysis": "",
            }

        if self.use_google_pro or self.use_bing_pro:
            relevant_info = extract_relevant_info_pro(results)[:self.top_k]
        else:
            relevant_info = extract_relevant_info(results)[:self.top_k]

        # Fetch all URLs in one batch
        urls_to_fetch = [doc_info['url'] for doc_info in relevant_info]

        contents = {}
        if urls_to_fetch:
            try:
                contents = await fetch_page_content_async(
                    urls_to_fetch,
                    use_jina=self.use_jina,
                    jina_api_key=self.jina_api_key,
                    keep_links=self.keep_links,
                    api_counters=self.api_counters,
                    max_concurrent=200,
                    show_progress=False,
                )
            except Exception as e:
                print(f"Error fetching URLs: {e}")

        for doc_info in relevant_info:
            url = doc_info['url']
            raw_content = contents.get(url, "")
                
            if raw_content:
                is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)

            has_error = any(indicator.lower() in raw_content.lower() for indicator in error_indicators) or raw_content == "" or raw_content.startswith("Error:")

            if has_error:
                doc_info['page_info'] = "Can not fetch the page content."
            else:
                doc_info['page_info'] = raw_content

        formatted_documents = format_search_results(relevant_info)

        try:
            class Args:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            args = Args(
                model_name=self.model_name,
                aux_model_name=self.aux_model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                repetition_penalty=self.repetition_penalty,
                top_k_sampling=self.top_k_sampling,
                min_p=self.min_p,
                use_jina=self.use_jina,
                jina_api_key=self.jina_api_key,
                keep_links=self.keep_links,
                use_bing_pro=self.use_bing_pro,
                bing_pro_token=self.bing_pro_token,
                bing_pro_api=self.bing_pro_api,
                use_google_pro=self.use_google_pro,
                google_pro_api_key=self.google_pro_api_key,
                use_custom_api=self.use_custom_api,
                custom_api_url=self.custom_api_url,
                use_aihubmix=self.use_aihubmix,
                aihubmix_api_url=self.aihubmix_api_url,
                aihubmix_api_keys=self.aihubmix_api_keys,
                api_key=self.api_key,
                aux_api_key=self.aux_api_key,
                top_k=self.top_k,
            )

            semaphore = asyncio.Semaphore(self.model_concurrent_limit)
            
            # Create empty caches for the deep web explorer since we're not using caching
            empty_search_cache = {}
            empty_url_cache = {}
            empty_cache_lock = asyncio.Lock()

            analysis, _ = await generate_deep_web_explorer(
                client=self.client,
                aux_client=self.aux_client,
                search_query=query,
                search_intent=search_intent,
                document=formatted_documents,
                args=args,
                search_cache=empty_search_cache,
                url_cache=empty_url_cache,
                search_cache_lock=empty_cache_lock,
                url_cache_lock=empty_cache_lock,
                semaphore=semaphore,
                model_semaphore=model_semaphore,
                bing_semaphore=bing_semaphore,
                aux_model_semaphore=aux_model_semaphore,
                api_counters=self.api_counters,
                use_aihubmix=args.use_aihubmix if hasattr(args, 'use_aihubmix') else False,
                aihubmix_api_url=args.aihubmix_api_url if hasattr(args, 'aihubmix_api_url') else None,
                aihubmix_api_keys=args.aihubmix_api_keys if hasattr(args, 'aihubmix_api_keys') else None,
            )

            extracted_info = extract_answer_fn(analysis, mode='summary')
            
            # Return all the search information
            return {
                "extracted_info": extracted_info,
                "search_query": query,
                "search_intent": search_intent,
                "formatted_documents": formatted_documents,
                "analysis": analysis,
            }
        except Exception as e:
            print(f"Error in deep web exploration: {e}")
            return {
                "extracted_info": f"Error in deep exploration: {str(e)}\n\nSearch Results:\n{formatted_documents}",
                "search_query": query,
                "search_intent": search_intent,
                "formatted_documents": formatted_documents,
                "analysis": "",
            }

    def search_intent_sync(self, query: str, intent: str) -> dict:
        """
        Synchronous version of the search_with_intent method.
        Args:
            query: The search query
            intent: The search intent (already generated)
        Returns:
            Dictionary containing extracted information and search details
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.search_with_intent(query, intent))
            loop.close()
            return result
        except Exception as e:
            print(f"Error executing search_with_intent: {str(e)}")
            return {
                "extracted_info": f"Error executing search_with_intent: {str(e)}",
                "search_query": query,
                "search_intent": intent,
                "formatted_documents": "",
                "analysis": "",
            }


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Web Search Tool and Verifier")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--pred_answer", type=str, help="Predicted answer to verify")
    parser.add_argument("--context", type=str, default="", help="Optional context for the search")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:8000", help="Base URL for the API endpoint")
    parser.add_argument("--aux_api_base_url", type=str, default="http://localhost:8000", help="Base URL for the auxiliary model API endpoint")
    parser.add_argument("--bing_subscription_key", type=str, help="Bing Search API subscription key")
    parser.add_argument("--use_bing_pro", action="store_true", help="Use Bing Search Pro API")
    parser.add_argument("--bing_pro_token", type=str, default="1791013312122257441", help="Token for Bing Search Pro API")
    parser.add_argument("--use_google_pro", action="store_true", help="Use Google Search Pro API")
    parser.add_argument("--google_pro_api_key", type=str, default="81b2d7ef2974da1a63669e7ffa5534a6974ff990", help="API key for Google Search Pro")
    parser.add_argument("--api_key", type=str, default="empty", help="API key for the main model")
    parser.add_argument("--aux_api_key", type=str, default="empty", help="API key for the auxiliary model")
    parser.add_argument("--use_custom_api", action="store_true", help="Whether to use custom API for generation")
    parser.add_argument("--custom_api_url", type=str, default=None, help="URL for custom API endpoint")
    parser.add_argument("--use_aihubmix", action="store_true", help="Whether to use AIHubMix API")
    parser.add_argument("--aihubmix_api_url", type=str, default="https://aihubmix.com/v1/chat/completions", help="URL for AIHubMix API endpoint")
    parser.add_argument("--aihubmix_api_keys", type=str, default="sk-7lwnQNrbElFFkHgUEbA2E12eE9944648BeDdDcB432D1C096", help="API key(s) for AIHubMix (comma-separated)")
    parser.add_argument("--model_name", type=str, default="QwQ-32B", help="Name of the main model to use")
    parser.add_argument("--aux_model_name", type=str, default="Qwen2.5-32B-Instruct", help="Name of the auxiliary model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.05, help="Minimum p sampling parameter")
    parser.add_argument("--top_k", type=int, default=10, help="Maximum number of search documents to return")
    parser.add_argument("--top_k_sampling", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--max_tokens", type=int, default=81920, help="Maximum number of tokens to generate")
    parser.add_argument("--input_path", type=str, default="", help="Path to input JSON file with questions/problems")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output results")
    parser.add_argument("--max_search_calls", type=int, default=15, help="Maximum number of search calls allowed")
    parser.add_argument("--concurrent_limit", type=int, default=32, help="Maximum number of concurrent processes")
    parser.add_argument("--mode", type=str, default="search", choices=["search", "solve", "verify", "verify_budget_forcing", "solve_budget_forcing"], 
                       help="Mode to run: search for web search, solve for solving problems, verify for verifying answers, verify_budget_forcing for budget-forced verification, solve_budget_forcing for budget-forced solving")
    parser.add_argument("--price_config_path", type=str, default="./model_config/aihubmix_price.json", 
                       help="Path to the price configuration file")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create search tool
    search_tool = WebSearchTool(
        api_base_url=args.api_base_url,
        aux_api_base_url=args.aux_api_base_url,
        model_name=args.model_name,
        aux_model_name=args.aux_model_name,
        bing_subscription_key=args.bing_subscription_key,
        use_bing_pro=args.use_bing_pro,
        bing_pro_token=args.bing_pro_token,
        use_google_pro=args.use_google_pro,
        google_pro_api_key=args.google_pro_api_key,
        api_key=args.api_key,
        aux_api_key=args.aux_api_key,
        use_custom_api=args.use_custom_api,
        custom_api_url=args.custom_api_url,
        use_aihubmix=args.use_aihubmix,
        aihubmix_api_url=args.aihubmix_api_url,
        aihubmix_api_keys=args.aihubmix_api_keys,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k_sampling=args.top_k_sampling,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens
    )
    
    
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
    
    # Common system messages for different modes
    solver_prompt = """You are a reasoning assistant equipped with web search capabilities to help solve problems. Use the "deep_websearch" tool whenever you need additional information or evidence. You may use this tool multiple times, up to a maximum of **{max_search_calls} times**.

Your answer must be provided within a boxed environment like this:

\\[
\\boxed{{<your final answer>}}
\\]"""

    verifier_prompt1 = """You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** the accuracy of a predicted answer to a user's question. Use the "deep_websearch" tool whenever you need additional evidence or verification. You may use this tool multiple times, up to a maximum of **{max_search_calls} times**.

Follow these steps to evaluate the answer:

1. Clearly list all conditions that the predicted answer must satisfy based on the user's question.
2. For each condition, verify whether it is satisfied by the predicted answer.
3. Calculate your confidence score as follows (assign a confidence score between 0 and 1):


Confidence Score = (number of satisfied conditions / total conditions)


Provide your final confidence score (float number) clearly in the following format:

\\[
\\boxed{{<confidence_score>}}
\\]"""

    verifier_prompt4 = """
You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** the accuracy of a predicted answer provided to a user's question. To ensure thorough verification, actively use the **"deep_websearch"** tool whenever you require additional evidence or confirmation. You may utilize this tool multiple times, up to a maximum of {max_search_calls} searches.

Follow these steps carefully:

1. Clearly list all conditions that must be satisfied by the predicted answer.
2. Use the **"deep_websearch"** tool to gather evidence verifying each condition as needed.
3. Calculate the confidence score (between 0 and 1) as the ratio of the number of conditions verified through web search to the total number of conditions.

Finally, provide the confidence score (float number) explicitly in the following format:

\\[
\\boxed{{\\text{{<confidence_score>}}}}
\\]

Here, `<confidence_score>` equals **(number of conditions verified using deep_websearch) / (total number of conditions)**.
"""


    verifier_prompt2 = """
You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** the accuracy of a predicted answer provided to a user's question. To ensure thorough verification, actively use the **"deep_websearch"** tool whenever you require additional evidence or confirmation. You may utilize this tool multiple times, up to a maximum of {max_search_calls} searches.

Follow these steps carefully:

1. Clearly list all conditions that must be satisfied by the predicted answer.
2. Use the **"deep_websearch"** tool to gather evidence verifying each condition as needed.
3. Calculate the confidence score (between 0 and 1) as the ratio of the number of conditions verified through web search to the total number of conditions. (Please note that if the predicted answer does not provide a clear answer to the question, directly assign a confidence score of 0)

Finally, provide the confidence score (float number) explicitly in the following format:

\\[
\\boxed{{\\text{{<confidence_score>}}}}
\\]

Here, `<confidence_score>` equals **(number of conditions verified using deep_websearch) / (total number of conditions)**.
"""

    verifier_prompt3 = '''
    You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** the accuracy of a predicted answer provided to a user's question. Use the "deep_websearch" tool whenever additional information or evidence is needed. You may use this tool multiple times, up to a maximum of **{max_search_calls} times**.

At the end, you must provide a confidence score indicating whether the predicted answer is the correct (ground truth) answer to the question. This score is calculated as the ratio of the number of conditions from the question that are verified through "deep_websearch" to the total number of conditions.

If the predicted answer fails to clearly address the question — for example, if no answer is provided — assign a confidence score of 0.

Report the confidence score (as a float) using the format:

\\[
\\boxed{{\\text{{<confidence_score>}}}}
\\]
    '''
    
    

    verifier_prompt5 = """
You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** the accuracy of a predicted answer provided to a user's question. To ensure rigorous and reliable verification, actively use the **"deep_websearch"** tool whenever you need supporting evidence. You may use this tool multiple times, up to a maximum of {max_search_calls} searches.

Follow these steps carefully:

1. Identify and clearly list **all conditions** that the predicted answer must satisfy based on the user's question.
2. For **each condition**, use the **"deep_websearch"** tool to gather evidence. A condition is considered **verified** only if:
   - It has been directly supported by a search result (i.e., verified through deep_websearch), **and**
   - The search result confirms that the predicted answer **does satisfy** that condition.
3. Compute the confidence score as the **ratio of the number of conditions that are both verified through deep_websearch and confirmed as satisfied** to the **total number of conditions**.

Finally, output the confidence score (as a float between 0 and 1) using the following format:

\\[
\\boxed{{\\text{{<confidence_score>}}}}
\\]

Here, `<confidence_score>` = **(number of conditions both verified and satisfied via deep_websearch) / (total number of conditions)**.
"""

    verifier_prompt = '''
    You are a reasoning assistant equipped with web search capabilities to fact-check and verify the accuracy of a predicted answer to a user's question. Your sole task is to verify the predicted answer, not to answer the question yourself.

To ensure rigorous and reliable verification, actively use the "deep_websearch" tool whenever supporting evidence is needed. You may use this tool multiple times, up to a maximum of {max_search_calls} searches.

Follow these steps carefully:

Carefully read the user's question and extract all conditions that a correct answer must satisfy. These are the requirements imposed by the question — not properties of the predicted answer alone.

For each condition, check whether the predicted answer satisfies it. Use deep_websearch to find evidence. A condition is considered verified only if:

It is explicitly supported by search results obtained through deep_websearch, and

The search results confirm that the predicted answer does indeed satisfy the condition.

Compute the confidence score as the ratio of conditions that are both (i) verified using deep_websearch and (ii) confirmed to be satisfied, to the total number of identified conditions.

Finally, present the confidence score in the following format:

\\[
\\boxed{{\\text{{<confidence_score>}}}}
\\]

Here, <confidence_score> = (number of conditions both verified and satisfied via deep_websearch) / (total number of conditions from the question).
    '''

    
    # Create the appropriate tool based on the mode
    if args.mode in ["solve", "verify", "solve_budget_forcing", "verify_budget_forcing"]:
        system_message = solver_prompt if args.mode == "solve" else verifier_prompt
        
        policy_tool = PolicyTool(
            api_base_url=args.api_base_url,
            aux_api_base_url=args.aux_api_base_url,
            model_name=args.model_name,
            system_message=system_message,
            aux_model_name=args.aux_model_name,
            api_key=args.api_key,
            aux_api_key=args.aux_api_key,
            tools=tools,
            tool_choice=tool_choice,
            max_search_calls=args.max_search_calls,
            use_custom_api=args.use_custom_api,
            custom_api_url=args.custom_api_url,
            use_aihubmix=args.use_aihubmix,
            aihubmix_api_url=args.aihubmix_api_url,
            aihubmix_api_keys=args.aihubmix_api_keys,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            price_config_path=args.price_config_path
        )
    
    # Check which mode we're running in
    if args.mode == "search" and args.query:
        # Single query search mode
        result = search_tool.search_sync(args.query, args.context)
        print("\n===== SEARCH RESULT =====")
        print(result)
        print("=========================")
        
    elif args.mode == "solve" and args.query:
        # Single problem solving mode
        answer, messages, search_results = policy_tool.solve_problem(args.query, search_tool, args.max_search_calls)
        print("\n===== SOLUTION =====")
        print(answer)
        print("====================")
        
        # Save statistics for single problem mode
        if policy_tool.use_aihubmix:
            policy_tool.save_statistics(args.output_dir, search_tool)
        
    elif args.mode == "verify" and args.query and args.pred_answer:
        # Single problem verification mode
        confidence, messages, search_results = policy_tool.verify_problem(args.query, args.pred_answer, search_tool, args.max_search_calls)
        print("\n===== VERIFICATION RESULT =====")
        print(f"Question: {args.query}")
        print(f"Predicted Answer: {args.pred_answer}")
        print(f"Confidence Score: {confidence}")
        print("===============================")
        
        # Save statistics for single verification mode
        if policy_tool.use_aihubmix:
            policy_tool.save_statistics(args.output_dir, search_tool)
    
    elif args.input_path:
        # Batch processing mode
        try:
            with open(args.input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            problems_list = []
            for item in data:
                # Copy the entire item to problems_list, preserving all original fields
                problem_dict = item.copy()
                # Ensure the question field exists
                if "question" not in problem_dict and "Question" in problem_dict:
                    problem_dict["question"] = problem_dict["Question"]
                problems_list.append(problem_dict)
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Process based on the mode
            if args.mode == "search":
                print("Batch search mode not implemented. Use solve or verify mode.")
            elif args.mode == "solve":
                # Process the batch of problems
                print(f"Solving {len(problems_list)} problems with max_search_calls={args.max_search_calls}, concurrent_limit={args.concurrent_limit}")
                policy_tool.process_batch_sync(
                    problems_list, 
                    search_tool, 
                    max_search_calls=args.max_search_calls, 
                    concurrent_limit=args.concurrent_limit, 
                    output_dir=args.output_dir
                )
            elif args.mode == "verify":
                # Process the batch of problems for verification
                print(f"Verifying {len(problems_list)} problems with max_search_calls={args.max_search_calls}, concurrent_limit={args.concurrent_limit}")
                policy_tool.verify_batch_sync(
                    problems_list, 
                    search_tool, 
                    max_search_calls=args.max_search_calls, 
                    concurrent_limit=args.concurrent_limit, 
                    output_dir=args.output_dir
                )
            elif args.mode == "verify_budget_forcing":
                # Process the batch of problems for budget forcing verification
                print(f"Verifying with budget forcing {len(problems_list)} problems with max_search_calls={args.max_search_calls}, concurrent_limit={args.concurrent_limit}")
                policy_tool.verify_batch_budget_forcing_sync(
                    problems_list, 
                    search_tool, 
                    max_search_calls=args.max_search_calls, 
                    concurrent_limit=args.concurrent_limit, 
                    output_dir=args.output_dir
                )
            elif args.mode == "solve_budget_forcing":
                # Process the batch of problems for budget forcing solving
                print(f"Solving with budget forcing {len(problems_list)} problems with max_search_calls={args.max_search_calls}, concurrent_limit={args.concurrent_limit}")
                policy_tool.solve_batch_budget_forcing_sync(
                    problems_list, 
                    search_tool, 
                    max_search_calls=args.max_search_calls, 
                    concurrent_limit=args.concurrent_limit, 
                    output_dir=args.output_dir
                )
            print(f"Results saved to {args.output_dir}")
        except Exception as e:
            print(f"Error processing batch: {e}")
    else:
        if args.mode == "search":
            print("Please specify --query for single query mode or --input_path for batch mode")
        elif args.mode == "solve":
            print("Please specify --query for single problem mode or --input_path for batch mode")
        elif args.mode == "verify":
            print("Please specify both --query and --pred_answer for single verification mode or --input_path for batch mode")
        elif args.mode == "verify_budget_forcing":
            print("Please specify --input_path for batch mode with budget forcing verification")
        elif args.mode == "solve_budget_forcing":
            print("Please specify --input_path for batch mode with budget forcing solving")
    
    # Print API counters statistics
    print("\n===== API COUNTERS =====")
    for counter_name, count in search_tool.api_counters.items():
        print(f"{counter_name}: {count}")
    print("=========================")
    
    # Print token usage statistics and cost calculation if AIHubMix API was used
    if args.mode in ["solve", "verify", "solve_budget_forcing", "verify_budget_forcing"] and policy_tool.use_aihubmix:
        print("\n===== TOTAL TOKEN USAGE =====")
        print(f"Total prompt tokens: {policy_tool.token_counters['prompt_tokens']}")
        print(f"Total completion tokens: {policy_tool.token_counters['completion_tokens']}")
        print(f"Total tokens: {policy_tool.token_counters['total_tokens']}")
        print("==============================")
        
        # Calculate and display cost information
        cost_info = policy_tool.calculate_cost()
        if cost_info:
            print("\n===== COST CALCULATION =====")
            print(f"Model used for pricing: {cost_info['model']}")
            print(f"Prompt price per 1K tokens: ${cost_info['pricing']['prompt_price_per_k']:.5f}")
            print(f"Completion price per 1K tokens: ${cost_info['pricing']['completion_price_per_k']:.5f}")
            print(f"Prompt cost: ${cost_info['prompt_cost']:.5f}")
            print(f"Completion cost: ${cost_info['completion_cost']:.5f}")
            print(f"Total cost: ${cost_info['total_cost']:.5f}")
            print("==============================")
            
        # For non-batch modes, statistics are already saved in the respective code sections