import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import os, time
from collections import defaultdict
# from lcb_runner.evaluation import codegen_metrics
import sys
sys.path.append('./scripts/utils')
from math_equivalence import is_equiv
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List
import requests
import random


def extract_answer_fn(output, mode='qa', extract_answer=False):
    if extract_answer == False and mode not in ['infogen', 'summary', 'research']:
        if mode == 'qa':
            return output.strip()
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer
    extracted_text = ''
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'  # Extract the code between ```python and ```
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode in ['infogen', 'summary', 'research']:
        pattern_info = "**Final Information"
        if "</think>\n" in output:
            extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        elif pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        else:
            # extracted_text = "No helpful information found."
            extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
        if mode == 'research':
            extracted_text = extracted_text[:6000]
        else:
            extracted_text = extracted_text[:2500]
    elif mode in ['math', 'choose', 'qa']:
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
        else:
            pattern = 'ANSWER:'
            if pattern in output:
                extracted_text = output.split(pattern)[-1].strip('**').strip()
        if mode in ['choose']:
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    return extracted_text


def get_random_key(api_key):
    """Get a random key from a comma-separated list of keys"""
    if api_key and ',' in api_key:
        keys = api_key.split(',')
        return random.choice(keys)
    return api_key


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
                
            # For other errors
            print(f"API request failed with status {response.status_code}: {response.text}")
            
        except Exception as e:
            print(f"Request error: {e}")
        
        # Exponential backoff
        sleep_time = (2 ** i) + random.random()
        print(f"Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
    
    return "Error: Failed to get response after multiple retries."


def _make_aihubmix_api_request(api_url, headers, json_data, aihubmix_api_keys=None, max_retries=200):
    """Helper function to make AIHubMix API requests with retries"""
    # Determine API keys to use
    api_keys = aihubmix_api_keys
    if api_keys and ',' in api_keys:
        keys = api_keys.split(',')
    else:
        keys = [api_keys]
    
    # Initialize provider settings
    providers = ["anthropic", "openai", "google", "mistral", "cohere", "meta", "qwen"]
    provider_settings = {}
    for provider in providers:
        provider_settings[provider] = {"enabled": True}
    
    for i in range(max_retries):
        # Select a random API key
        current_key = random.choice(keys)
        
        try:
            # Set up the request headers with the current key
            current_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {current_key}"
            }
            
            # Add provider settings to the request
            current_json_data = json_data.copy()
            current_json_data["provider_settings"] = provider_settings
            
            # Make the request
            response = requests.post(
                api_url,
                headers=current_headers,
                json=current_json_data,
                timeout=60
            )
            
            if response.status_code == 200:
                try:
                    res = json.loads(response.text)
                    return res['choices'][0]['message']['content']
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error parsing AIHubMix API response: {e}")
                    print(f"Response text: {response.text}")
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                print(f"AIHubMix rate limited. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                continue
                
            # For other errors
            print(f"AIHubMix API request failed with status {response.status_code}: {response.text}")
            
        except Exception as e:
            print(f"AIHubMix request error: {e}")
        
        # Exponential backoff
        sleep_time = (2 ** min(i, 10)) + random.random()  # Cap the exponent to avoid extremely long waits
        print(f"Retrying AIHubMix in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
    
    return "Error: Failed to get response from AIHubMix after multiple retries."


async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
    extract_answer: bool = False,
    use_custom_api: bool = False,
    custom_api_url: str = None,
    api_key: str = "empty",
    use_aihubmix: bool = False,
    aihubmix_api_url: str = None,
    aihubmix_api_keys: str = None,
) -> bool:
    """Evaluate a single pair of answers using LLM"""

    if extract_answer:
        prompt = f"""You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Predicted Answer: {pred_answer}

Are these answers equivalent? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""
    else:
        prompt = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                if use_aihubmix and aihubmix_api_url:
                    # Use AIHubMix API
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {aihubmix_api_keys}"
                    }
                    
                    json_data = {
                        "model": model_name, 
                        "messages": [{"role": "user", "content": prompt}], 
                        "temperature": 0.0,  # Use deterministic response for evaluation
                    }
                    
                    # Make synchronous request with asyncio
                    response_text = await asyncio.to_thread(
                        _make_aihubmix_api_request,
                        aihubmix_api_url,
                        headers,
                        json_data,
                        aihubmix_api_keys
                    )
                    
                    llm_judge = pred_answer != '' and (is_equiv(pred_answer, labeled_answer) or \
                        response_text.lower() == "correct" and \
                        not ("incorrect" in response_text.lower() or \
                             "wrong" in response_text.lower() or \
                             "not mention" in pred_answer.lower() or \
                             "no definitive answer" in pred_answer.lower() or \
                             "insufficient" in pred_answer.lower() or \
                             "no such article" in pred_answer.lower() or \
                             "no match" in pred_answer.lower() or \
                             "unknown" in pred_answer.lower() or \
                             "cannot be determined" in pred_answer.lower() or \
                             "no article match" in pred_answer.lower() or \
                             "not enough information" in pred_answer.lower() or \
                             "not stated" in pred_answer.lower() or \
                             "not specified" in pred_answer.lower() or \
                             "not available" in pred_answer.lower() or \
                             "meets all" in pred_answer.lower() or \
                             "no publicky documented artist" in pred_answer.lower() or \
                             "no publicly known artist" in pred_answer.lower() or \
                             "no publicly" in pred_answer.lower() or \
                             "no such" in pred_answer.lower() or \
                             "no known" in pred_answer.lower() or \
                             "no specific" in pred_answer.lower() or \
                             "not enough" in pred_answer.lower() or \
                             "available information" in pred_answer.lower() or \
                             "matches all" in pred_answer.lower() or \
                             "no verifiable" in pred_answer.lower() or \
                             "unable to determine" in pred_answer.lower() or \
                             "no conclusive" in pred_answer.lower() or \
                             "unknown" in pred_answer.lower() or \
                             "no definitive" in pred_answer.lower() or \
                             "no exact" in pred_answer.lower() or \
                             "no answer" in pred_answer.lower() or \
                             "not publicly" in pred_answer.lower() or \
                             "no available" in pred_answer.lower() or \
                             "not found" in pred_answer.lower() or \
                             "not found in" in pred_answer.lower() or \
                             "not correct" in pred_answer.lower()))
                    return llm_judge, response_text
                elif use_custom_api and custom_api_url:
                    # Use the custom API
                    current_api_key = get_random_key(api_key)
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": current_api_key
                    }
                    
                    json_data = {
                        "model": model_name, 
                        "messages": [{"role": "user", "content": prompt}], 
                        "temperature": 0.0,  # Use deterministic response for evaluation
                    }
                    
                    # Make synchronous request with asyncio
                    response_text = await asyncio.to_thread(
                        _make_custom_api_request,
                        custom_api_url,
                        headers,
                        json_data
                    )
                    
                    llm_judge = pred_answer != '' and (is_equiv(pred_answer, labeled_answer) or \
                        response_text.lower() == "correct" and \
                        not ("incorrect" in response_text.lower() or \
                             "wrong" in response_text.lower() or \
                             "not mention" in pred_answer.lower() or \
                             "no definitive answer" in pred_answer.lower() or \
                             "insufficient" in pred_answer.lower() or \
                             "no such article" in pred_answer.lower() or \
                             "no match" in pred_answer.lower() or \
                             "unknown" in pred_answer.lower() or \
                             "cannot be determined" in pred_answer.lower() or \
                             "no article match" in pred_answer.lower() or \
                             "not enough information" in pred_answer.lower() or \
                             "not stated" in pred_answer.lower() or \
                             "not specified" in pred_answer.lower() or \
                             "not available" in pred_answer.lower() or \
                             "meets all" in pred_answer.lower() or \
                             "no publicky documented artist" in pred_answer.lower() or \
                             "no publicly known artist" in pred_answer.lower() or \
                             "no publicly" in pred_answer.lower() or \
                             "no such" in pred_answer.lower() or \
                             "no known" in pred_answer.lower() or \
                             "no specific" in pred_answer.lower() or \
                             "not enough" in pred_answer.lower() or \
                             "available information" in pred_answer.lower() or \
                             "matches all" in pred_answer.lower() or \
                             "no verifiable" in pred_answer.lower() or \
                             "unable to determine" in pred_answer.lower() or \
                             "no conclusive" in pred_answer.lower() or \
                             "unknown" in pred_answer.lower() or \
                             "no definitive" in pred_answer.lower() or \
                             "no exact" in pred_answer.lower() or \
                             "no answer" in pred_answer.lower() or \
                             "not publicly" in pred_answer.lower() or \
                             "no available" in pred_answer.lower() or \
                             "not found" in pred_answer.lower() or \
                             "not found in" in pred_answer.lower() or \
                             "not correct" in pred_answer.lower()))
                    return llm_judge, response_text
                else:
                    # Use standard AsyncOpenAI client
                    chat_response = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    response_text = chat_response.choices[0].message.content.strip()
                    llm_judge = pred_answer != '' and (is_equiv(pred_answer, labeled_answer) or \
                        response_text.lower() == "correct" and \
                        not ("incorrect" in response_text.lower() or \
                             "wrong" in response_text.lower() or \
                             "not correct" in response_text.lower()))
                    return llm_judge, response_text
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                return is_equiv(pred_answer, labeled_answer), "Error"
            await asyncio.sleep(1 * (attempt + 1))
    
    return is_equiv(pred_answer, labeled_answer), "Error"


async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
    extract_answer: bool = False,
    use_custom_api: bool = False,
    custom_api_url: str = None,
    use_aihubmix: bool = False,
    aihubmix_api_url: str = None,
    aihubmix_api_keys: str = None,
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = None
    if model_name is None:
        model_name = "Qwen2.5-72B-Instruct"

    client = None
    if not use_custom_api and not use_aihubmix:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )

    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
            extract_answer=extract_answer,
            use_custom_api=use_custom_api,
            custom_api_url=custom_api_url,
            api_key=api_key,
            use_aihubmix=use_aihubmix,
            aihubmix_api_url=aihubmix_api_url,
            aihubmix_api_keys=aihubmix_api_keys
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)
    
    return results


def evaluate_predictions(output, labeled_answer, mode='math', use_llm=False, question=None, extract_answer=False):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'llm_equal': 0}
    pred_answer = extract_answer_fn(output, mode=mode, extract_answer=extract_answer)
    pred_answer_new = pred_answer
    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    else:
        # If no answer was extracted, keep only the last 3 lines
        pred_answer_new = '\n'.join(output.replace("\n\n", "\n").strip().split('\n')[-5:])

    if mode in ['qa']:
        def normalize_answer_qa(s):
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.strip().split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        normalized_pred_answer = normalize_answer_qa(pred_answer_new)

        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    elif mode in ['math', 'choose']:
        def normalize_answer(text):
            text = text.lower()
            text = " ".join(text.strip().split())
            return text
        normalized_pred_answer = normalize_answer(pred_answer_new)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)
        
        # Add LLM-based evaluation if requested
        if use_llm and question is not None:
            final_metric["llm_equal"] = 0  # Will be updated in batch later

    return final_metric, pred_answer


def run_evaluation(filtered_data, input_list, output_list, task_type, output_dir, output_metrics_path, output_metrics_overall_path, use_llm=False, extract_answer=False, domain_fields=None, api_base_url=None, model_name=None, use_custom_api=False, custom_api_url=None, api_key=None, use_aihubmix=False, aihubmix_api_url=None, aihubmix_api_keys=None):
    # Initialize domain metrics dictionary
    domain_metrics = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'em': [],
        'acc': [],
        'f1': [],
        'math_equal': [],
        'llm_equal': [],
        'pass@1': []
    })

    # Helper function to get domain from item
    def get_domain(item):
        for field in domain_fields:
            if field in item and item[field] is not None:
                return item[field]
        return 'Unknown'

    if task_type == 'code':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []
        num_valid_answer = 0

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            # 检查是否已经有预测答案字段
            if 'pred_answer' in item:
                # 如果已经有预测答案，直接使用
                pred_code = item['pred_answer']
                item['Pred_Answer'] = pred_code
                item['Question'] = input_prompt
            else:
                # 如果没有预测答案，从输出中提取
                if type(result) == str:
                    item['Output'] = result
                else:
                    item['Output'] = result.outputs[0].text

                if item['Output'] == '':
                    item['Pred_Answer'] = ''
                    item['Question'] = input_prompt
                    item['Metrics'] = {'pass@1': 0}
                    item['Results'] = {}
                    item['Final_metadata'] = {}
                    continue

                pred_code = extract_answer_fn(item['Output'], mode='codegen', extract_answer=extract_answer)
                item['Pred_Answer'] = pred_code

            if pred_code != '':
                num_valid_answer += 1

            public_test_cases = json.loads(item.get("test_cases", "{}"))

            inputs = public_test_cases.get("inputs", [])
            outputs = public_test_cases.get("outputs", [])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            if 'Question' not in item:
                item['Question'] = input_prompt

        # # Call codegen_metrics with pass@1
        # metrics, results, final_metadata = codegen_metrics(
        #     samples_list,
        #     generations_list,
        #     k_list=[1],  # Evaluate the top 1 generated result
        #     num_process_evaluate=10,   # Parallel evaluation
        #     timeout=10,  # Set timeout to 10 seconds
        #     debug=False,  # Enable debug mode
        # )

        # # Extract pass@1
        # pass_at_1 = metrics.get('pass@1', 0.0)
        # detail_pass_at_1 = metrics['detail']['pass@1']

        # for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
        #     item['Metrics'] = {'pass@1': pass1}
        #     item['Results'] = res
        #     item['Final_metadata'] = meta

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': 0.0, # pass_at_1,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        }

        # Add domain-specific metrics collection
        for item in filtered_data:
            domain = get_domain(item)
            domain_metrics[domain]['total'] += 1
            domain_metrics[domain]['pass@1'].append(0.0)

    elif task_type in ['math', 'choose', 'qa']:
        # Evaluation for math/qa tasks
        avg_em, avg_acc, avg_f1, avg_math, avg_llm = [], [], [], [], []
        num_valid_answer = 0
        
        # Lists to store data for batch LLM evaluation
        questions_for_llm = []
        labeled_answers_for_llm = []
        pred_answers_for_llm = []
        items_for_llm = []

        for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list), total=len(input_list)):
            # 检查是否已经有预测答案字段
            if 'pred_answer' in item:
                # 如果已经有预测答案，直接使用
                pred_answer = item['pred_answer']
                item['Pred_Answer'] = pred_answer
                item['Question'] = input_prompt
            else:
                # 如果没有预测答案，从输出中提取
                if type(result) == str:
                    item['Output'] = result
                else:
                    item['Output'] = result.outputs[0].text

                if item['Output'] == '':
                    item['Pred_Answer'] = ''
                    item['Question'] = input_prompt
                    item['Metrics'] = {
                        'em': 0,
                        'acc': 0,
                        'f1': 0,
                        'math_equal': 0,
                        'llm_equal': 0 if use_llm else None
                    }
                    avg_em.append(0)
                    avg_acc.append(0)
                    avg_f1.append(0)
                    avg_math.append(0)
                    if use_llm:
                        avg_llm.append(0)
                    continue

                # Get the labeled answer from the item
                labeled_answer = item.get('answer', '')  # Use get() to safely access the answer field
                if 'Correct Choice' in item and item['Correct Choice'] is not None:
                    labeled_answer = item['Correct Choice']
                elif 'answer_letter' in item and item['answer_letter'] is not None:
                    labeled_answer = item['answer_letter']
                metric, pred_answer = evaluate_predictions(
                    output=result, 
                    labeled_answer=labeled_answer,
                    mode=task_type,
                    use_llm=use_llm,
                    question=input_prompt,
                    extract_answer=extract_answer
                )
                
                item['Pred_Answer'] = pred_answer
                item['Metrics'] = metric
                item['Question'] = input_prompt

            # 无论预测答案是从哪来的，都需要进行评估
            # 获取标准答案
            labeled_answer = item.get('answer', '')  # Use get() to safely access the answer field
            if 'Correct Choice' in item and item['Correct Choice'] is not None:
                labeled_answer = item['Correct Choice']
            elif 'answer_letter' in item and item['answer_letter'] is not None:
                labeled_answer = item['answer_letter']

            # 如果是从pred_answer中获取的，需要计算评估指标
            if 'pred_answer' in item and 'Metrics' not in item:
                if "Output" in item:
                    # 如果没有输出，使用默认度量
                    item['Metrics'] = {
                        'em': 0,
                        'acc': 0,
                        'f1': 0,
                        'math_equal': 0,
                        'llm_equal': 0 if use_llm else None
                    }
                else:
                    # 为预定义的pred_answer创建一个虚拟输出以便计算度量
                    fake_output = item['pred_answer']
                    metric, _ = evaluate_predictions(
                        output=fake_output,
                        labeled_answer=labeled_answer,
                        mode=task_type,
                        use_llm=use_llm,
                        question=input_prompt,
                        extract_answer=False  # 不需要提取，直接使用
                    )
                    item['Metrics'] = metric

            # Store data for batch LLM evaluation
            if use_llm:
                questions_for_llm.append(input_prompt)
                labeled_answers_for_llm.append(labeled_answer)
                pred_answers_for_llm.append(item['Pred_Answer'])
                items_for_llm.append(item)

            # Determine the validity of the predicted answer
            my_method_valid = (item['Pred_Answer'] != '')

            avg_em.append(item['Metrics']['em'])
            avg_acc.append(item['Metrics']['acc'])
            avg_f1.append(item['Metrics']['f1'])
            avg_math.append(item['Metrics']['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

        # Perform batch LLM evaluation if needed
        if use_llm and questions_for_llm:
            llm_results = asyncio.run(llm_evaluate_equivalence_batch(
                questions=questions_for_llm,
                labeled_answers=labeled_answers_for_llm,
                pred_answers=pred_answers_for_llm,
                extract_answer=extract_answer,
                api_base_url=api_base_url,
                model_name=model_name,
                api_key=api_key,
                use_custom_api=use_custom_api,
                custom_api_url=custom_api_url,
                use_aihubmix=use_aihubmix,
                aihubmix_api_url=aihubmix_api_url,
                aihubmix_api_keys=aihubmix_api_keys,
            ))
            
            # Update metrics with LLM results
            for item, (llm_result, llm_response) in zip(items_for_llm, llm_results):
                item['Metrics']['llm_equal'] = int(llm_result)
                item['Metrics']['llm_response'] = llm_response
                avg_llm.append(int(llm_result))

        # Compute overall metrics
        overall_metrics = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_math) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        }
        
        # Add LLM evaluation metric if available
        if len(avg_llm) > 0:
            overall_metrics['llm_equal'] = np.mean(avg_llm)

        for item, metric in zip(filtered_data, [item['Metrics'] for item in filtered_data]):
            domain = get_domain(item)
            domain_metrics[domain]['total'] += 1
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])
            if 'llm_equal' in metric:
                domain_metrics[domain]['llm_equal'].append(metric['llm_equal'])

    # After the main evaluation loop and before saving metrics, add:
    # Calculate domain-specific metrics
    domain_metrics_final = {}
    for domain, metrics in domain_metrics.items():
        domain_metrics_final[domain] = {
            'total': metrics['total'],
            'em': np.mean(metrics['em']) if len(metrics['em']) > 0 else 0.0,
            'acc': np.mean(metrics['acc']) if len(metrics['acc']) > 0 else 0.0,
            'f1': np.mean(metrics['f1']) if len(metrics['f1']) > 0 else 0.0,
            'math_equal': np.mean(metrics['math_equal']) if len(metrics['math_equal']) > 0 else 0.0,
        }
        if metrics['llm_equal']:
            domain_metrics_final[domain]['llm_equal'] = np.mean(metrics['llm_equal'])
        if metrics['pass@1']:
            domain_metrics_final[domain]['pass@1'] = np.mean(metrics['pass@1'])

    # Add domain metrics to overall metrics
    overall_metrics['domain_metrics'] = domain_metrics_final
    
    print(overall_metrics)
    
    # Save prediction results and metrics
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model outputs.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--task', type=str, required=True, choices=['code', 'math', 'choose', 'qa', 'llm'], help='Task type for evaluation')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM for equivalence evaluation')
    parser.add_argument('--extract_answer', action='store_true', help='Extract answer from output')
    parser.add_argument('--api_base_url', type=str, default=None, help='Base URL for LLM API')
    parser.add_argument('--model_name', type=str, default=None, help='Model name for LLM evaluation')
    parser.add_argument('--api_key', type=str, default="empty", help='API key for LLM evaluation')
    parser.add_argument('--use_custom_api', action='store_true', help='Whether to use custom API for evaluation')
    parser.add_argument('--custom_api_url', type=str, default=None, help='URL for custom API endpoint')
    parser.add_argument('--use_aihubmix', action='store_true', help='Whether to use AIHubMix API')
    parser.add_argument('--aihubmix_api_url', type=str, default="https://aihubmix.com/v1/chat/completions", help='URL for AIHubMix API endpoint')
    parser.add_argument('--aihubmix_api_keys', type=str, default="sk-7lwnQNrbElFFkHgUEbA2E12eE9944648BeDdDcB432D1C096", help='API key(s) for AIHubMix (comma-separated)')
    args = parser.parse_args()

    # Define the list of domain field names to check (in order of priority)
    DOMAIN_FIELDS = ['Level', 'level', 'category', 'High-level domain', 'difficulty_level', 'field', 'problem_topic']

    output_path = args.output_path
    output_metrics_path = output_path.replace('.json', '.metrics.json')
    output_metrics_overall_path = output_path.replace('.json', '.metrics.overall.json')

    # Load main output data
    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare input_list and output_list for run_evaluation
    input_list = []
    output_list = []
    filtered_data = []
    
    if isinstance(data, dict):
        # Convert dict to list if data is a dictionary
        for key, item in data.items():
            if isinstance(item, dict):  # Ensure item is a dictionary
                filtered_data.append(item)
                input_list.append(item.get('question', item.get('Question', '')))
                # 优先使用pred_answer字段，如果不存在则使用result
                if 'pred_answer' in item:
                    # 如果有预测答案，创建一个虚拟输出以保持一致性
                    output_list.append(item.get('pred_answer'))
                else:
                    output_list.append(item.get('result'))
    else:
        # If data is already a list
        filtered_data = data
        for item in data:
            input_list.append(item.get('Question', item.get('question', '')))
            # 优先使用pred_answer字段，如果不存在则使用Output或result
            if 'pred_answer' in item:
                output_list.append(item.get('pred_answer'))
            else:
                output_list.append(item.get('Output', item.get('result')))

    # Run evaluation with domain fields
    run_evaluation(
        filtered_data=filtered_data,  # Pass the properly structured data
        input_list=input_list,
        output_list=output_list,
        task_type=args.task,
        output_dir=output_path,
        output_metrics_path=output_metrics_path,
        output_metrics_overall_path=output_metrics_overall_path,
        use_llm=args.use_llm,
        api_base_url=args.api_base_url,
        model_name=args.model_name,
        extract_answer=args.extract_answer,
        domain_fields=DOMAIN_FIELDS,  # Pass the domain fields to run_evaluation
        use_custom_api=args.use_custom_api,
        custom_api_url=args.custom_api_url,
        api_key=args.api_key,
        use_aihubmix=args.use_aihubmix,
        aihubmix_api_url=args.aihubmix_api_url,
        aihubmix_api_keys=args.aihubmix_api_keys,
    )

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
