import aiohttp
import asyncio
import re
import json
from typing import Tuple, List, Dict
from bing_search import (
    extract_relevant_info, 
    fetch_page_content_async,
    extract_snippet_with_context,
    bing_web_search_async
)
from utils import extract_answer_fn
from openai import AsyncOpenAI
from prompts import get_multiqa_search_o1_instruction, get_task_instruction_openqa, get_search_intent_instruction, get_deep_web_explorer_instruction, get_click_intent_instruction, get_web_page_reader_instruction
from settings import Environment


def prepare_init_prompt(query, env):
    instruction = get_multiqa_search_o1_instruction(env.max_search_limit)
    user_prompt = get_task_instruction_openqa(query)

    prompt = instruction + user_prompt
    prompt = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n'

    env.prompt = prompt
    env.prompt_tokens = len(prompt.split())
    return env,prompt


def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
    matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
    if matches:
        return matches[0][::-1].strip()
    return None

def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search reEND_SEARCH_QUERYdable string"""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>','').replace('</b>','')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
    return formatted_documents


async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    model_name: str = "QwQ-32B",
    stop: List[str] = ["<|end_search_query|>"],
    retry_limit: int = 3,
):
    """Generate a streaming response with retry logic"""
    for attempt in range(retry_limit):
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                extra_body={
                    'top_k': top_k,
                    'include_stop_str_in_output': True,
                    'repetition_penalty': repetition_penalty,
                    # 'min_p': min_p
                },
                timeout=3600,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].text:
                    yield chunk.choices[0].text
            return  

        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
            await asyncio.sleep(0.5 * (attempt + 1))
    
    yield ""  



async def get_search_result(env, search_query, search_intent):
    yield f'\n\nBegin searching for {search_query}......\n\n'

    if search_query in env.search_cache:
        results = env.search_cache[search_query]
    else:
        try:
            results = await bing_web_search_async(search_query, env.bing_subscription_key, env.bing_endpoint)
            env.search_cache[search_query] = results
        except Exception as e:
            print(f"Error during search query '{search_query}': {e}")
            results = {}
    #yield '\n\nSearch result: ' + str(results) + '\n\n'
    if 'webPages' in results and 'value' in results['webPages']:
        results['webPages']['value'] = results['webPages']['value'][:env.search_num]
        for item in results['webPages']['value']:
            if 'name' in item:
                item['name'] = item['name'].replace('<b>','').replace('</b>','')

        yield f"""Get {len(results['webPages']['value'])} web pages:\n\n"""
        yield '\n\n'.join([f"""[{item.get('name', '')}]({item.get('url', '')})""" for item in results['webPages']['value']]) + '\n\n'
    else:
        yield 'No relevant information found.\n\n'

    relevant_info = extract_relevant_info(results)[:env.search_num]
    urls_to_fetch = []
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in env.url_cache:
            urls_to_fetch.append(url)

    if urls_to_fetch:
        try:
            yield 'Browsing web pages...\n\n'
            contents = await fetch_page_content_async(
                urls_to_fetch, 
                use_jina=env.use_jina, 
                jina_api_key=env.jina_api_key, 
                keep_links=env.keep_links
            )
            for url, content in contents.items():
                # Only cache content if it doesn't contain error indicators
                has_error = (any(indicator.lower() in content.lower() for indicator in env.error_indicators) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
                if not has_error:
                    env.url_cache[url] = content
        except Exception as e:
            print(f"Error fetching URLs: {e}")

    # Get web page information for each result
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in env.url_cache:
            raw_content = ""
        else:
            raw_content = env.url_cache[url]
            is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=5000)

        # Check if content has error indicators
        has_error = any(indicator.lower() in raw_content.lower() for indicator in env.error_indicators) or raw_content == ""
    
        if has_error:
            # If content has error, use it directly as summary
            doc_info['page_info'] = "Can not fetch the page content."
        else:
            # Use raw content directly as page info
            doc_info['page_info'] = raw_content
    yield 'Reading completed!\n\n'
    formatted_documents = format_search_results(relevant_info)
    yield formatted_documents

async def generate_deep_web_explorer(
    env,
    search_query: str,
    search_intent: str,
    document: str,
):
    prompt = get_deep_web_explorer_instruction(search_query=search_query, search_intent=search_intent, search_result=document)
    prompt = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n'
    
    finished = False
    sub_env = env.add_child_env()
    sub_env.prompt = prompt

    while True:
        # Generate next response
        prompt = sub_env.prompt
        new_step = ''
        async for chunk in generate_response(
            client=env.client,
            prompt=prompt,
            temperature=env.temperature,
            top_p=env.top_p,
            max_tokens=env.max_tokens,
            repetition_penalty=env.repetition_penalty,
            top_k=env.top_k,
            min_p=env.min_p,
            model_name=env.use_model_name,
            stop=[env.END_SEARCH_QUERY, env.END_CLICK_LINK],
        ):
            yield True, chunk.replace('</think>','')
            new_step += chunk
        new_step = new_step.replace('</think>\n','')

        sub_env.update_step(new_step)

        if sub_env.total_tokens >= env.max_path_tokens or sub_env.interation_times >= env.max_interation_times:
            break

        # Check for search query
        if new_step.rstrip().endswith(env.END_SEARCH_QUERY):
            new_query = extract_between(new_step, env.BEGIN_SEARCH_QUERY, env.END_SEARCH_QUERY)
            if new_query:
                yield True, f'Begin searching for {new_query}......\n\n'
                if new_query in sub_env.executed_search_queries:
                    search_result = f"\n{env.BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{env.END_SEARCH_RESULT}\n"
                    sub_env.update_step(search_result)
                    yield True, 'The query has been searched before, use previous result.\n\n'
                    continue

                sub_env.update_search(new_query)
                
                # Execute search
                if new_query in sub_env.search_cache:
                    results = sub_env.search_cache[new_query]
                else:
                    try:
                        results = await bing_web_search_async(new_query, sub_env.bing_subscription_key, sub_env.bing_endpoint)
                        sub_env.search_cache[new_query] = results
                    except Exception as e:
                        print(f"Error during search query '{new_query}': {e}")
                        results = {}

                if 'webPages' in results and 'value' in results['webPages']:
                    results['webPages']['value'] = results['webPages']['value'][:sub_env.search_num]
                    for item in results['webPages']['value']:
                        if 'name' in item:
                            item['name'] = item['name'].replace('<b>','').replace('</b>','')
                    yield True, f"""Get {len(results['webPages']['value'])} web pages:\n\n"""
                    yield True, '\n\n'.join([f"""- [{item.get('name', '')}]({item.get('url', '')})""" for item in results['webPages']['value']]) + '\n\n'
                else:
                    yield True, 'No relevant information found.\n\n'


                relevant_info = extract_relevant_info(results)[:sub_env.search_num]

                formatted_documents = format_search_results(relevant_info)
                
                # Append search results
                search_result = f"\n{env.BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{env.END_SEARCH_RESULT}\n"
                sub_env.update_step(search_result)
                
        # Check for click link
        elif new_step.rstrip().endswith(env.END_CLICK_LINK):
            url = extract_between(new_step, env.BEGIN_CLICK_LINK, env.END_CLICK_LINK)
            yield True, f'\n\nBegin clicking the link: {url}...\n\n'
            prompt = get_click_intent_instruction(sub_env.output)
            prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
            click_intent = ''
            async for chunk in generate_response(
                client=env.aux_client,
                model_name=env.aux_model_name,
                prompt=prompt,
            ):
                click_intent += chunk

            if url and click_intent:
                if url in sub_env.clicked_urls:
                    # If URL was already clicked, append message
                    click_result = f"\n{env.BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{env.END_CLICK_RESULT}\nOK, let me use the previously found information."
                    sub_env.update_step(click_result)
                    yield True, 'The URL has been clicked before, use previous result.\n\n'
                    continue

                sub_env.update_click(url)  # Add URL to clicked set

                # Fetch and process page content
                if url not in sub_env.url_cache:
                    try:
                        content = await fetch_page_content_async(
                            [url], 
                            use_jina=env.use_jina, 
                            jina_api_key=env.jina_api_key, 
                            keep_links=env.keep_links
                        )
                        content = content[url]
                        # Only cache content if it doesn't contain error indicators
                        has_error = (any(indicator.lower() in content.lower() for indicator in env.error_indicators) and len(content.split()) < 64) or content == ''
                        if not has_error:
                            env.url_cache[url] = content
                    except Exception as e:
                        print(f"Error fetching URL {url}: {e}")
                        content = ""
                else:
                    content = env.url_cache[url]

                # Check if content has error indicators
                has_error = any(indicator.lower() in content.lower() for indicator in env.error_indicators) or content == ''
                
                if has_error:
                    # If content has error, use it directly as summary
                    summary = "Unable to fetch the page content. You can try other links."
                else:
                    # Use web page reader to summarize content
                    reader_prompt = get_web_page_reader_instruction(click_intent, content)
                    reader_prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{reader_prompt}<|im_end|>\n<|im_start|>assistant\n'

                    summary = await generate_response(
                        client=env.aux_client,
                        prompt=reader_prompt,
                        max_tokens=3600,
                        model_name=env.aux_model_name,
                    )

                # Append click results
                click_result = f"\n{env.BEGIN_CLICK_RESULT}\n{summary}\n{env.END_CLICK_RESULT}\n"
                yield True, 'I have read the relevant information of the web page.\n\n'
                sub_env.update_step(click_result)
        else:
            finished = True
            break
    
    # Add max limit message if needed
    if not finished and (sub_env.total_tokens >= env.max_path_tokens or sub_env.interation_times >= env.max_interation_times):
        output = f"\n{env.BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{env.END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        sub_env.update_step(output)
        final_response = ''
        async for chunk in generate_response(
            client=env.client,
            prompt=prompt,
            temperature=env.temperature,
            top_p=env.top_p,
            max_tokens=512,
            repetition_penalty=1.2,
            top_k=env.top_k,
            min_p=env.min_p,
            model_name=env.use_model_name,
        ):
            yield True, chunk
            final_response += chunk
        sub_env.update_step(final_response)
    yield False, sub_env.output




async def run_search_chain(env, new_step):
    print("in search chain")
    search_query = extract_between(new_step, env.BEGIN_SEARCH_QUERY, env.END_SEARCH_QUERY)
    if search_query is None or len(search_query) <= 5: # 太短了，不合法的query
        yield False, 'Current search query is too short, skip'
    else:
        if search_query in env.executed_search_queries:
            append_text = f"\n\n{env.BEGIN_SEARCH_RESULT}You have already searched for this query.{env.END_SEARCH_RESULT}\n\nOK, let me use the previously found information."
            yield False, append_text
        else:
            input_prompt = get_search_intent_instruction(env.output)
            input_prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_prompt}<|im_end|>\n<|im_start|>assistant\n'
            search_intent = ''
            async for chunk in generate_response(
                client=env.aux_client,
                model_name=env.aux_model_name,
                prompt=input_prompt,
            ):
                search_intent += chunk
            
            async for chunk in get_search_result(env, search_query, search_intent):
                if '***Web Page' not in chunk:
                    yield True, chunk
                else:
                    formatted_documents = chunk
            
            #yield 'Current search result: ' + formatted_documents
            async for (flag,chunk) in generate_deep_web_explorer(
                env,
                search_query=search_query,
                search_intent=search_intent,
                document=formatted_documents,
            ):
                yield flag, chunk

            analysis = chunk
            env.update_search(search_query)
            extracted_info = extract_answer_fn(analysis, mode='summary')
            # Update sequence with search results
            append_text = f"\n\n{env.BEGIN_SEARCH_RESULT}{extracted_info}{env.END_SEARCH_RESULT}\n\n"
            yield False, append_text


async def process_query_async(query, env):
    env, prompt = prepare_init_prompt(query, env)
    while True:
        prompt = env.prompt
        collected_step = ""
        async for text_chunk in generate_response(
            client=env.client, 
            prompt=prompt, 
            temperature=env.temperature, 
            top_p=env.top_p, 
            max_tokens=env.max_tokens, 
            repetition_penalty=env.repetition_penalty, 
            top_k=env.top_k, 
            min_p=env.min_p, 
            model_name=env.use_model_name, 
            stop=[env.END_SEARCH_QUERY]
        ):
            collected_step += text_chunk
            yield text_chunk.replace('</think>','')
        new_step = collected_step.replace('</think>\n', '')
        env.update_step(new_step)

        if not new_step.endswith(env.END_SEARCH_QUERY):
            break

        if env.search_count >= env.max_search_limit or env.total_tokens >= env.max_path_tokens:
            append_text = f"\n\n{env.BEGIN_SEARCH_RESULT}You have reached the search limit. You are not allowed to search.{env.END_SEARCH_RESULT}\n\n"
        else:
            async for (flag, chunk) in run_search_chain(env, new_step):
                if flag:
                    yield chunk
            append_text = chunk
            
        if append_text != '':
            env.update_step(append_text)
            
if __name__ == "__main__":
    env = Environment()
    asyncio.run(process_query_async("List all presidents of the United States", env))
