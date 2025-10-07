import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union
from urllib.parse import urljoin
import aiohttp
import asyncio
import chardet
import random
import nltk
from collections import Counter

# Set custom NLTK data path and download required data
nltk_data_path = '/ssddata/wzengak/browse-tts/nltk'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download punkt tokenizer data if it's not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    print("Downloading punkt tokenizer data...")
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)


# ----------------------- Set your WebParserClient URL -----------------------
WebParserClient_url = None


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)

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

class WebParserClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化Web解析器客户端
        
        Args:
            base_url: API服务器的基础URL，默认为本地测试服务器
        """
        self.base_url = base_url.rstrip('/')
        
    def parse_urls(self, urls: List[str], timeout: int = 120) -> List[Dict[str, Union[str, bool]]]:
        """
        发送URL列表到解析服务器并获取解析结果
        
        Args:
            urls: 需要解析的URL列表
            timeout: 请求超时时间，默认20秒
            
        Returns:
            解析结果列表
            
        Raises:
            requests.exceptions.RequestException: 当API请求失败时
            requests.exceptions.Timeout: 当请求超时时
        """
        endpoint = urljoin(self.base_url, "/parse_urls")
        response = requests.post(endpoint, json={"urls": urls}, timeout=timeout)
        response.raise_for_status()  # 如果响应状态码不是200，抛出异常
        
        return response.json()["results"]


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 3000) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:100000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            # if end_index - start_index < 2 * context_chars:
            #     end_index = min(len(full_text), start_index + 2 * context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None, keep_links=False):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippet (Optional[str]): The snippet to search for.
        keep_links (bool): Whether to keep links in the extracted text.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            if 'pdf' in url:
                return extract_pdf_text(url)

            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                
                # 添加编码检测和处理
                if response.encoding.lower() == 'iso-8859-1':
                    # 尝试从内容检测正确的编码
                    response.encoding = response.apparent_encoding
                
                try:
                    soup = BeautifulSoup(response.text, 'lxml')
                except Exception:
                    soup = BeautifulSoup(response.text, 'html.parser')

                # Check if content has error indicators
                has_error = (any(indicator.lower() in response.text.lower() for indicator in error_indicators) and len(response.text.split()) < 64) or response.text == ''
                if has_error:
                    if WebParserClient_url is None:
                        # If WebParserClient is not available, return error message
                        return f"Error extracting content: {str(e)}"
                    # If content has error, use WebParserClient as fallback
                    client = WebParserClient(WebParserClient_url)
                    results = client.parse_urls([url])
                    if results and results[0]["success"]:
                        text = results[0]["content"]
                    else:
                        error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                        return f"WebParserClient error: {error_msg}"
                else:
                    if keep_links:
                        # Clean and extract main content
                        # Remove script, style tags etc
                        for element in soup.find_all(['script', 'style', 'meta', 'link']):
                            element.decompose()

                        # Extract text and links
                        text_parts = []
                        for element in soup.body.descendants if soup.body else soup.descendants:
                            if isinstance(element, str) and element.strip():
                                # Clean extra whitespace
                                cleaned_text = ' '.join(element.strip().split())
                                if cleaned_text:
                                    text_parts.append(cleaned_text)
                            elif element.name == 'a' and element.get('href'):
                                href = element.get('href')
                                link_text = element.get_text(strip=True)
                                if href and link_text:  # Only process a tags with both text and href
                                    # Handle relative URLs
                                    if href.startswith('/'):
                                        base_url = '/'.join(url.split('/')[:3])
                                        href = base_url + href
                                    elif not href.startswith(('http://', 'https://')):
                                        href = url.rstrip('/') + '/' + href
                                    text_parts.append(f"[{link_text}]({href})")

                        # Merge text with reasonable spacing
                        text = ' '.join(text_parts)
                        # Clean extra spaces
                        text = ' '.join(text.split())
                    else:
                        text = soup.get_text(separator=' ', strip=True)
            except Exception as e:
                if WebParserClient_url is None:
                    # If WebParserClient is not available, return error message
                    return f"Error extracting content: {str(e)}"
                # If normal extraction fails, try using WebParserClient
                client = WebParserClient(WebParserClient_url)
                results = client.parse_urls([url])
                if results and results[0]["success"]:
                    text = results[0]["content"]
                else:
                    error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                    return f"WebParserClient error: {error_msg}"

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:20000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None, show_progress=False, keep_links=False):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.
        show_progress (bool): Whether to show progress bar with tqdm.
        keep_links (bool): Whether to keep links in the extracted text.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None, keep_links): url
            for url in urls
        }
        completed_futures = concurrent.futures.as_completed(futures)
        if show_progress:
            completed_futures = tqdm(completed_futures, desc="Fetching URLs", total=len(urls))
            
        for future in completed_futures:
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            # time.sleep(0.1)  # Simple rate limiting
    return results

def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            return search_results
        except Timeout:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Timeout occurred, retrying ({retry_count}/{max_retries})...")
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
        time.sleep(1)  # Wait 1 second between retries
    
    return {}  # Should never reach here but added for completeness


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = full_text
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    
    return useful_info




async def bing_web_search_async(query, subscription_key, endpoint, market='en-US', language='en', timeout=20, semaphore=None, api_counters=None):
    """
    Perform an asynchronous search using the Bing Web Search API.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int): Request timeout in seconds.
        semaphore (asyncio.Semaphore, optional): Semaphore for limiting concurrent requests.
        api_counters (Counter, optional): Counter object to track API calls.

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Use semaphore if provided
            if semaphore:
                async with semaphore:
                    # Increment API counter if provided
                    if api_counters is not None:
                        api_counters['bing_search'] += 1
                    
                    response = session.get(endpoint, headers=headers, params=params, timeout=timeout)
                    response.raise_for_status()
                    search_results = response.json()
                    return search_results
            else:
                # Increment API counter if provided
                if api_counters is not None:
                    api_counters['bing_search'] += 1
                
                response = session.get(endpoint, headers=headers, params=params, timeout=timeout)
                response.raise_for_status()
                search_results = response.json()
                return search_results
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
            time.sleep(1)  # Wait 1 second between retries

    return {}

class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int = 60):
        """
        初始化速率限制器
        
        Args:
            rate_limit: 在时间窗口内允许的最大请求数
            time_window: 时间窗口大小(秒)，默认60秒
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取一个令牌，如果没有可用令牌则等待"""
        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window)
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(random.randint(5, 30))  # 等待xxx秒后重试
            
            self.tokens -= 1
            return True

# 创建全局速率限制器实例
jina_rate_limiter = RateLimiter(rate_limit=130)  # 每分钟xxx次，避免报错

async def extract_text_from_url_async(url: str, session: aiohttp.ClientSession, use_jina: bool = False, 
                                    jina_api_key: Optional[str] = None, snippet: Optional[str] = None, 
                                    keep_links: bool = False) -> str:
    """Async version of extract_text_from_url"""
    try:
        if use_jina:
            # 在调用jina之前获取令牌
            await jina_rate_limiter.acquire()
            
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            async with session.get(f'https://r.jina.ai/{url}', headers=jina_headers) as response:
                text = await response.text()
                if not keep_links:
                    pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                    text = re.sub(pattern, "", text)
                text = text.replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            if 'pdf' in url:
                # Use async PDF handling
                text = await extract_pdf_text_async(url, session)
                return text[:10000]

            try:
                async with session.get(url) as response:
                    # 检测和处理编码
                    content_type = response.headers.get('content-type', '').lower()
                    if 'charset' in content_type:
                        charset = content_type.split('charset=')[-1]
                        html = await response.text(encoding=charset)
                    else:
                        # 如果没有指定编码，先用bytes读取内容
                        try:
                            content = await response.read()
                            # 使用chardet检测编码
                            detected = chardet.detect(content)
                            encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                            html = content.decode(encoding, errors='replace')
                        except asyncio.CancelledError:
                            # Handle cancellation during read operation
                            print(f"Request cancelled while reading content for URL: {url}")
                            return f"Error: Request cancelled while fetching {url}"
                    
                    # 检查是否有错误指示
                    has_error = (any(indicator.lower() in html.lower() for indicator in error_indicators) and len(html.split()) < 64) or len(html) < 50 or len(html.split()) < 20
                    # has_error = len(html.split()) < 64
                    if has_error:
                        if WebParserClient_url is None:
                            # If WebParserClient is not available, return error message
                            return f"Error: Content too short or contains error indicators for {url}"
                        # If content has error, use WebParserClient as fallback
                        client = WebParserClient(WebParserClient_url)
                        results = client.parse_urls([url])
                        if results and results[0]["success"]:
                            text = results[0]["content"]
                        else:
                            error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                            return f"WebParserClient error: {error_msg}"
                    else:
                        try:
                            soup = BeautifulSoup(html, 'lxml')
                        except Exception:
                            soup = BeautifulSoup(html, 'html.parser')

                        if keep_links:
                            # Similar link handling logic as in synchronous version
                            for element in soup.find_all(['script', 'style', 'meta', 'link']):
                                element.decompose()

                            text_parts = []
                            for element in soup.body.descendants if soup.body else soup.descendants:
                                if isinstance(element, str) and element.strip():
                                    cleaned_text = ' '.join(element.strip().split())
                                    if cleaned_text:
                                        text_parts.append(cleaned_text)
                                elif element.name == 'a' and element.get('href'):
                                    href = element.get('href')
                                    link_text = element.get_text(strip=True)
                                    if href and link_text:
                                        if href.startswith('/'):
                                            base_url = '/'.join(url.split('/')[:3])
                                            href = base_url + href
                                        elif not href.startswith(('http://', 'https://')):
                                            href = url.rstrip('/') + '/' + href
                                        text_parts.append(f"[{link_text}]({href})")

                            text = ' '.join(text_parts)
                            text = ' '.join(text.split())
                        else:
                            text = soup.get_text(separator=' ', strip=True)
            except asyncio.CancelledError:
                # Handle cancellation during request
                print(f"Request cancelled for URL: {url}")
                return f"Error: Request cancelled while fetching {url}"
            except aiohttp.ClientError as e:
                # Handle other client errors
                print(f"Client error for URL {url}: {str(e)}")
                return f"Error: Client error while fetching {url}: {str(e)}"

        # print('---\n', text[:1000])
        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return context if success else text
        else:
            return text[:50000]

    except asyncio.CancelledError:
        # Catch CancelledError at the top level
        print(f"Operation cancelled for URL: {url}")
        return f"Error: Operation cancelled while processing {url}"
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return f"Error fetching {url}: {str(e)}"

async def fetch_page_content_async(urls: List[str], use_jina: bool = False, jina_api_key: Optional[str] = None, 
                                 snippets: Optional[Dict[str, str]] = None, show_progress: bool = False,
                                 keep_links: bool = False, max_concurrent: int = 32, api_counters: Optional[Counter] = None) -> Dict[str, str]:
    """Asynchronously fetch content from multiple URLs."""
    async def process_urls():
        # Set proxy using environment variables
        # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
        # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
        # os.environ['no_proxy'] = '10.238.0.188,10.238.41.81,localhost,127.0.0.1'
        
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=240)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout, 
            headers=headers,
            trust_env=True  # Use environment variables for proxy
        ) as session:
            tasks = []
            for url in urls:
                # Increment API counter for each URL to be fetched
                if api_counters is not None:
                    api_counters['page_fetch'] += 1

                    
                task = extract_text_from_url_async(
                    url, 
                    session, 
                    use_jina, 
                    jina_api_key,
                    snippets.get(url) if snippets else None,
                    keep_links
                )
                tasks.append(task)
            
            if show_progress:
                results = []
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching URLs"):
                    try:
                        result = await task
                        results.append(result)
                    except asyncio.CancelledError:
                        print(f"Task was cancelled during processing")
                        results.append("Error: Task was cancelled")
                    except Exception as e:
                        print(f"Error in task: {str(e)}")
                        results.append(f"Error: {str(e)}")
            else:
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Handle any exceptions in the results
                    processed_results = []
                    for result in results:
                        if isinstance(result, Exception):
                            if isinstance(result, asyncio.CancelledError):
                                processed_results.append("Error: Task was cancelled")
                            else:
                                processed_results.append(f"Error: {str(result)}")
                        else:
                            processed_results.append(result)
                    results = processed_results
                except asyncio.CancelledError:
                    print("URL fetching was cancelled")
                    return {url: "Error: Operation was cancelled" for url in urls}
            
            return {url: result for url, result in zip(urls, results)}

    try:
        return await process_urls()
    except asyncio.CancelledError:
        print("All URL processing was cancelled")
        return {url: "Error: All operations were cancelled" for url in urls}

async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously extract text from a PDF.

    Args:
        url (str): URL of the PDF file.
        session (aiohttp.ClientSession): Aiohttp client session.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        async with session.get(url, timeout=30) as response:  # Set timeout to 20 seconds
            if response.status != 200:
                return f"Error: Unable to retrieve the PDF (status code {response.status})"
            
            content = await response.read()
            
            # Open the PDF file using pdfplumber
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
            
            # Limit the text length
            cleaned_text = full_text
            return cleaned_text
    except asyncio.TimeoutError:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def get_random_key(api_key):
    """Get a random key from a comma-separated list of keys"""
    if api_key and ',' in api_key:
        keys = api_key.split(',')
        return random.choice(keys)
    return api_key

def bing_web_search_pro(query, token="1791013312122257441", api="bing-search-pro", max_retries=5):
    """
    Perform a search query using the Friday API's Bing Search Pro.
    
    Args:
        query (str): The search query.
        token (str): Authorization token or comma-separated list of tokens.
        api (str): The API type to use.
        max_retries (int): Maximum number of retry attempts.
        
    Returns:
        dict: The response data from the API.
    """
    url = "https://aigc.sankuai.com/v1/friday/api/search"
    
    data = {
        "query": query,
        "api": api
    }
    
    for attempt in range(max_retries):
        # Select a new random token on each attempt
        current_token = get_random_key(token)
        
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - Attempt {attempt+1}/{max_retries} with token {current_token}")
                print(response.text)
                # Add delay between retries with some randomness
                time.sleep(1 + random.random())
        except Exception as e:
            print(f"Exception during request (attempt {attempt+1}/{max_retries} with token {current_token}): {e}")
            time.sleep(2 + random.random())
    
    return {"error": f"Failed after {max_retries} attempts"}

async def bing_web_search_async_pro(query, token="1791013312122257441", api="bing-search-pro", max_retries=200, semaphore=None, api_counters=None):
    """
    Perform an asynchronous search query using the Friday API's Bing Search Pro.
    
    Args:
        query (str): The search query.
        token (str): Authorization token or comma-separated list of tokens.
        api (str): The API type to use.
        max_retries (int): Maximum number of retry attempts.
        semaphore (asyncio.Semaphore, optional): Semaphore for limiting concurrent requests.
        api_counters (Counter, optional): Counter object to track API calls.
        
    Returns:
        dict: The response data from the API.
    """
    url = "https://aigc.sankuai.com/v1/friday/api/search"
    
    data = {
        "query": query,
        "api": api
    }
    print("-----current query:----- ", query, "-----current token:----- ", token)
    # Create a rate limiter for the Friday API similar to the jina_rate_limiter
    friday_rate_limiter = RateLimiter(rate_limit=10)  # Adjust rate limit as needed
    
    # # Make sure environment variables are set for proxy
    # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['no_proxy'] = '10.238.0.188,10.238.41.81,localhost,127.0.0.1'
    
    for attempt in range(max_retries):
        # Select a new random token on each attempt
        current_token = get_random_key(token)
        
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Content-Type": "application/json"
        }
        
        try:
            # Acquire token from rate limiter before making request
            await friday_rate_limiter.acquire()
            
            # Increment API counter if provided
            if api_counters is not None:
                api_counters['bing_search_pro'] += 1
            
            # Use semaphore if provided
            if semaphore:
                async with semaphore:
                    async with aiohttp.ClientSession(trust_env=True) as session:
                        async with session.post(
                            url=url,
                            headers=headers,
                            json=data,
                            timeout=30
                        ) as response:
                            if response.status == 200:
                                return await response.json()
                            else:
                                response_text = await response.text()
                                print(f"Error: {response.status} - Attempt {attempt+1}/{max_retries} with token {current_token}")
                                print(response_text)
                                # Add delay between retries with some randomness
                                await asyncio.sleep(1 + random.random())
            else:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.post(
                        url=url,
                        headers=headers,
                        json=data,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            response_text = await response.text()
                            print(f"Error: {response.status} - Attempt {attempt+1}/{max_retries} with token {current_token}")
                            print(response_text)
                            # Add delay between retries with some randomness
                            await asyncio.sleep(1 + random.random())
        except Exception as e:
            print(f"Exception during request (attempt {attempt+1}/{max_retries} with token {current_token}): {e}")
            await asyncio.sleep(2 + random.random())
    
    return {"error": f"Failed after {max_retries} attempts"}

def extract_relevant_info_pro(search_results):
    """
    Extract relevant information from Bing Search Pro (Friday API) results, Tencent Search results,
    or Google Serper API results.

    Args:
        search_results (dict): JSON response from the search function.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    # Handle Google Serper API results
    if "organic" in search_results and isinstance(search_results["organic"], list):
        # Extract organic search results
        organic_results = search_results.get("organic", [])
        
        # Parse the search results
        for id, item in enumerate(organic_results):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'site_name': item.get('domain', '') or (item.get('link', '').split('/')[2] if item.get('link') else ''),
                'date': item.get('date', ''),  # Google Serper sometimes returns dates
                'snippet': item.get('snippet', ''),
                'context': ''  # Reserved field to be filled later
            }
            
            # Add position if available
            if "position" in item:
                info["position"] = item.get("position")
            
            # Add sitelinks if available
            if "sitelinks" in item:
                info["sitelinks"] = item.get("sitelinks")
                
            # Add thumbnail if available
            if "thumbnailUrl" in item:
                info["thumbnail"] = item.get("thumbnailUrl")
            
            useful_info.append(info)
        
        # Add knowledge graph information if available
        if "knowledgeGraph" in search_results:
            kg = search_results["knowledgeGraph"]
            kg_info = {
                'id': 0,  # Give knowledge graph priority
                'title': kg.get('title', ''),
                'url': kg.get('descriptionLink', ''),
                'site_name': kg.get('descriptionSource', ''),
                'date': '',
                'snippet': kg.get('description', ''),
                'context': '',  # Reserved field to be filled later
                'is_knowledge_graph': True,
                'type': kg.get('type', ''),
                'attributes': kg.get('attributes', {})
            }
            useful_info.insert(0, kg_info)  # Insert at the beginning
            
        return useful_info
    
    # Check if response is valid for Bing/Tencent
    if not search_results or search_results.get("code") != "200":
        if "error" in search_results:
            print(f"Error in search results: {search_results['error']}")
        return useful_info
    
    # Handle Tencent Search results
    if "tencentSearchResults" in search_results:
        tencent_results = search_results.get("tencentSearchResults", {})
        pages = tencent_results.get("Response", {}).get("Pages", [])
        
        for id, page_str in enumerate(pages):
            try:
                # Parse the JSON string into a dictionary
                page = json.loads(page_str)
                
                info = {
                    'id': id + 1,  # Increment id for easier subsequent operations
                    'title': page.get('title', ''),
                    'url': page.get('url', ''),
                    'site_name': page.get('site', '') or (page.get('url', '').split('/')[2] if page.get('url') else ''),
                    'date': page.get('date', '').split()[0] if page.get('date') else '',
                    'snippet': page.get('passage', ''),
                    'context': ''  # Reserved field to be filled later
                }
                
                # Add thumbnail if available
                if "images" in page and page["images"]:
                    info["thumbnail"] = page["images"][0]
                
                useful_info.append(info)
            except json.JSONDecodeError as e:
                print(f"Error parsing page JSON: {e}")
                continue
        
        return useful_info
    
    # Handle Bing Search Pro results
    bing_results = search_results.get("bingSearchProResults", {})
    
    # Get query information
    query_context = bing_results.get("queryContext", {})
    original_query = query_context.get("originalQuery")
    
    # Check if we have web pages results
    web_pages = bing_results.get("webPages", {})
    if not web_pages:
        return useful_info
    
    # Parse the search results
    for id, item in enumerate(web_pages.get("value", [])):
        info = {
            'id': id + 1,  # Increment id for easier subsequent operations
            'title': item.get('name', ''),
            'url': item.get('url', ''),
            'site_name': item.get('displayUrl', '').split('/')[0] if 'displayUrl' in item else '',
            'date': item.get('datePublished', '').split('T')[0] if 'datePublished' in item else '',
            'snippet': item.get('snippet', ''),
            'context': ''  # Reserved field to be filled later
        }
        
        # Add thumbnail if available
        if "thumbnailUrl" in item:
            info["thumbnail"] = item.get("thumbnailUrl")
        
        # Add rating information if available
        if "about" in item and item["about"] and "aggregateRating" in item["about"][0]:
            rating = item["about"][0]["aggregateRating"]
            info["rating"] = {
                "value": rating.get("ratingValue"),
                "count": rating.get("reviewCount"),
                "max": rating.get("bestRating")
            }
        
        useful_info.append(info)
    
    return useful_info

def google_web_search_pro(query, api_key="81b2d7ef2974da1a63669e7ffa5534a6974ff990", max_retries=5, num_results=10):
    """
    Perform a search query using the Google Serper API.
    
    Args:
        query (str): The search query.
        api_key (str): API key for Google Serper or comma-separated list of keys.
        max_retries (int): Maximum number of retry attempts.
        num_results (int): Number of search results to return.
        
    Returns:
        dict: The response data from the API.
    """
    url = "https://google.serper.dev/search"
        # Make sure environment variables are set for proxy
    # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['no_proxy'] = '10.238.0.188,10.238.41.81,localhost,127.0.0.1'
    
    data = {
        "q": query,
        "num": num_results
    }
    
    for attempt in range(max_retries):
        # Select a new random key on each attempt
        current_key = get_random_key(api_key)
        
        headers = {
            "X-API-KEY": current_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - Attempt {attempt+1}/{max_retries} with key {current_key}")
                print(response.text)
                # Add delay between retries with some randomness
                time.sleep(1 + random.random())
        except Exception as e:
            print(f"Exception during request (attempt {attempt+1}/{max_retries} with key {current_key}): {e}")
            time.sleep(2 + random.random())
    
    return {"error": f"Failed after {max_retries} attempts"}

async def google_web_search_async_pro(query, api_key="81b2d7ef2974da1a63669e7ffa5534a6974ff990", max_retries=200, num_results=10, semaphore=None, api_counters=None):
    """
    Perform an asynchronous search query using the Google Serper API.
    
    Args:
        query (str): The search query.
        api_key (str): API key for Google Serper or comma-separated list of keys.
        max_retries (int): Maximum number of retry attempts.
        num_results (int): Number of search results to return.
        semaphore (asyncio.Semaphore, optional): Semaphore for limiting concurrent requests.
        api_counters (Counter, optional): Counter object to track API calls.
        
    Returns:
        dict: The response data from the API.
    """
    url = "https://google.serper.dev/search"
    
    data = {
        "q": query,
        "num": num_results
    }
    
    print("-----current query:----- ", query, "-----current api_key:----- ", api_key)
    
    # Create a rate limiter for the Google Serper API
    google_rate_limiter = RateLimiter(rate_limit=10)  # Adjust rate limit as needed
    
    # Make sure environment variables are set for proxy
    # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['no_proxy'] = '10.238.0.188,10.238.41.81,localhost,127.0.0.1'
    
    for attempt in range(max_retries):
        # Select a new random key on each attempt
        current_key = get_random_key(api_key)
        
        headers = {
            "X-API-KEY": current_key,
            "Content-Type": "application/json"
        }
        
        try:
            # Acquire token from rate limiter before making request
            await google_rate_limiter.acquire()
            
            # Increment API counter if provided
            if api_counters is not None:
                api_counters['google_search_pro'] += 1
            
            # Use semaphore if provided
            if semaphore:
                async with semaphore:
                    async with aiohttp.ClientSession(trust_env=True) as session:
                        async with session.post(
                            url=url,
                            headers=headers,
                            json=data,
                            timeout=30
                        ) as response:
                            if response.status == 200:
                                return await response.json()
                            else:
                                response_text = await response.text()
                                print(f"Error: {response.status} - Attempt {attempt+1}/{max_retries} with key {current_key}")
                                print(response_text)
                                # Add delay between retries with some randomness
                                await asyncio.sleep(1 + random.random())
            else:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.post(
                        url=url,
                        headers=headers,
                        json=data,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            response_text = await response.text()
                            print(f"Error: {response.status} - Attempt {attempt+1}/{max_retries} with key {current_key}")
                            print(response_text)
                            # Add delay between retries with some randomness
                            await asyncio.sleep(1 + random.random())
        except Exception as e:
            print(f"Exception during request (attempt {attempt+1}/{max_retries} with key {current_key}): {e}")
            await asyncio.sleep(2 + random.random())
    
    return {"error": f"Failed after {max_retries} attempts"}

def extract_google_search_results(search_results):
    """
    Extract relevant information from Google Serper API search results.

    Args:
        search_results (dict): JSON response from the google_web_search_pro function.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    # Check if response is valid
    if not search_results or "error" in search_results:
        if "error" in search_results:
            print(f"Error in search results: {search_results['error']}")
        return useful_info
    
    # Extract organic search results
    organic_results = search_results.get("organic", [])
    
    # Parse the search results
    for id, item in enumerate(organic_results):
        info = {
            'id': id + 1,  # Increment id for easier subsequent operations
            'title': item.get('title', ''),
            'url': item.get('link', ''),
            'site_name': item.get('domain', '') or (item.get('link', '').split('/')[2] if item.get('link') else ''),
            'date': '',  # Google Serper doesn't typically return dates
            'snippet': item.get('snippet', ''),
            'context': ''  # Reserved field to be filled later
        }
        
        # Add position if available
        if "position" in item:
            info["position"] = item.get("position")
        
        # Add sitelinks if available
        if "sitelinks" in item:
            info["sitelinks"] = item.get("sitelinks")
            
        # Add thumbnail if available
        if "thumbnailUrl" in item:
            info["thumbnail"] = item.get("thumbnailUrl")
        
        useful_info.append(info)
    
    return useful_info

async def fetch_page_content_turbo(urls: List[str], use_jina: bool = False, jina_api_key: Optional[str] = None,
                                snippets: Optional[Dict[str, str]] = None, show_progress: bool = False,
                                keep_links: bool = False, batch_size: int = 50, 
                                api_counters: Optional[Counter] = None) -> Dict[str, str]:
    """
    Turbo-charged asynchronous content fetching with aggressive optimization for maximum speed.
    
    Optimizations include:
    - Batched processing with dynamic worker scaling
    - HTTP/2 support when available
    - Aggressive connection pooling and reuse
    - Chunk-based content processing
    - Optimized timeout handling
    - Fast failure for problematic URLs
    
    Args:
        urls (List[str]): List of URLs to process
        use_jina (bool): Whether to use Jina for extraction
        jina_api_key (Optional[str]): API key for Jina
        snippets (Optional[Dict[str, str]]): A dictionary mapping URLs to their respective snippets
        show_progress (bool): Whether to show progress bar
        keep_links (bool): Whether to keep links in the extracted text
        batch_size (int): Size of URL batches for processing
        api_counters (Optional[Counter]): Counter object to track API calls
        
    Returns:
        Dict[str, str]: A dictionary mapping URLs to their extracted content
    """
    # Set proxy using environment variables
    # os.environ['http_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['https_proxy'] = 'http://10.253.34.172:6666'
    # os.environ['no_proxy'] = '10.238.0.188,10.238.41.81,localhost,127.0.0.1'

    # Create connection for TCP with optimal settings
    connector = aiohttp.TCPConnector(
        limit=0,  # No limit on connections per host
        force_close=False,  # Keep connections open
        enable_cleanup_closed=True,  # Clean up closed connections
        ssl=False,  # Skip SSL verification for speed
        use_dns_cache=True,  # Enable DNS caching
        ttl_dns_cache=300,  # Cache DNS results for 5 minutes
    )
    
    # More aggressive timeout settings
    timeout_config = aiohttp.ClientTimeout(
        total=120,        # Total timeout
        connect=10,       # Connection timeout
        sock_connect=10,  # Socket connection timeout
        sock_read=30      # Socket read timeout
    )
    
    # Optimize for large number of URLs
    optimal_workers = min(300, len(urls) * 2)  # More aggressive worker scaling
    
    # Results dictionary with thread safety
    from concurrent.futures import ThreadPoolExecutor
    import threading
    results = {}
    results_lock = threading.Lock()
    
    # Prepare URL batches
    url_batches = [urls[i:i+batch_size] for i in range(0, len(urls), batch_size)]
    
    # Progress tracking
    completed = 0
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(urls), desc="Turbo Fetching")
    
    # Semaphore to control concurrency
    sem = asyncio.Semaphore(optimal_workers)

    # Fast HTML parser setup - preload and optimize
    from bs4 import BeautifulSoup
    try:
        from lxml import etree
        parser = 'lxml'
    except ImportError:
        parser = 'html.parser'
    
    # Function to handle HTML parsing in a separate thread to avoid blocking event loop
    def parse_html(html_content):
        try:
            soup = BeautifulSoup(html_content, parser)
            # Remove unnecessary elements to speed up parsing
            for element in soup.find_all(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer']):
                element.decompose()
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            return f"Error parsing HTML: {str(e)}"
    
    # Thread pool for CPU-bound tasks
    cpu_executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4 if os.cpu_count() else 8))
    
    # Extraction optimization
    async def optimized_extract(url, session):
        # Skip if URL is too long or malformed (quick early rejection)
        if len(url) > 500 or not (url.startswith('http://') or url.startswith('https://')):
            return f"Error: Invalid URL format: {url}"
            
        # Increment API counter if provided
        if api_counters is not None:
            api_counters['page_fetch'] += 1
            
        # Quick check for PDF - handle differently
        if 'pdf' in url.lower():
            return await extract_pdf_text_async(url, session)
        
        # Handle Jina specifically
        if use_jina:
            try:
                await jina_rate_limiter.acquire()
                jina_headers = {
                    'Authorization': f'Bearer {jina_api_key}',
                    'X-Return-Format': 'markdown',
                }
                async with session.get(f'https://r.jina.ai/{url}', headers=jina_headers) as response:
                    if response.status != 200:
                        return f"Jina error: HTTP {response.status}"
                    text = await response.text()
                    if not keep_links:
                        pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                        text = re.sub(pattern, "", text)
                    return text.replace('---','-').replace('===','=').replace('   ',' ')
            except Exception as e:
                return f"Jina extraction error: {str(e)}"
        
        # Standard extraction with optimizations
        try:
            # Make request with optimized settings
            async with sem, session.get(url, allow_redirects=True, 
                                      timeout=timeout_config, 
                                      raise_for_status=False) as response:
                # Quick status check - fail fast
                if response.status >= 400:
                    return f"HTTP error: {response.status}"

                # For binary content types, don't process further
                content_type = response.headers.get('content-type', '').lower()
                if content_type and ('image/' in content_type or 'video/' in content_type or 'audio/' in content_type):
                    return f"Skipped binary content: {content_type}"
                
                # Check content length - don't download huge pages
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 10_000_000:  # 10MB limit
                    return f"Content too large: {content_length} bytes"
                
                # Read content in chunks to avoid memory issues
                max_content_size = 5_000_000  # 5MB
                content = b""
                total_read = 0
                
                async for chunk in response.content.iter_chunked(1024 * 64):  # 64KB chunks
                    content += chunk
                    total_read += len(chunk)
                    if total_read > max_content_size:
                        break
                
                # Detect encoding
                encoding = 'utf-8'  # Default
                try:
                    detected = chardet.detect(content[:20000])  # Only use first 20KB for detection
                    if detected['encoding'] and detected['confidence'] > 0.7:
                        encoding = detected['encoding']
                except Exception:
                    pass  # Fall back to default encoding
                
                try:
                    html = content.decode(encoding, errors='replace')
                    
                    # Skip pages with error indicators
                    has_error = any(indicator.lower() in html.lower()[:10000] for indicator in error_indicators)
                    if has_error:
                        if WebParserClient_url is not None:
                            # Use WebParserClient as fallback
                            client = WebParserClient(WebParserClient_url)
                            results = client.parse_urls([url])
                            if results and results[0]["success"]:
                                return results[0]["content"]
                            return f"WebParserClient error: {results[0].get('error', 'Unknown')}"
                        return "Error indicators found in content"
                    
                    # Use thread pool for CPU-bound HTML parsing
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(cpu_executor, parse_html, html)
                    
                    # Get context if snippet is provided
                    if snippet := snippets.get(url):
                        success, context = extract_snippet_with_context(text, snippet)
                        return context if success else text[:20000]
                    else:
                        return text[:20000]  # Limit to 20K chars
                
                except Exception as e:
                    return f"Content processing error: {str(e)}"
                    
        except asyncio.TimeoutError:
            return "Error: Request timed out"
        except Exception as e:
            return f"Request error: {str(e)}"
    
    async def process_batch(batch):
        nonlocal completed
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            headers=headers,
            trust_env=True,
            raise_for_status=False
        ) as session:
            tasks = [optimized_extract(url, session) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            with results_lock:
                for url, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results[url] = f"Error: {str(result)}"
                    else:
                        results[url] = result
                    
                    # Update progress
                    completed += 1
                    if pbar:
                        pbar.update(1)
    
    # Main processing logic - handle both sync and async contexts
    try:
        # Process batches concurrently
        batch_tasks = [process_batch(batch) for batch in url_batches]
        
        # Properly handle different event loop contexts
        try:
            # Check if we're in an event loop already
            loop = asyncio.get_running_loop()
            # If we get here, we're already in an async context
            # Just await all batches
            await asyncio.gather(*batch_tasks)
        except RuntimeError:
            # No running event loop, create a new one
            asyncio.run(asyncio.gather(*batch_tasks))
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
    finally:
        if pbar:
            pbar.close()
        cpu_executor.shutdown(wait=False)
    
    return results

# Create a synchronous wrapper for the async function for compatibility
def fetch_page_content_turbo_sync(urls: List[str], use_jina: bool = False, jina_api_key: Optional[str] = None,
                               snippets: Optional[Dict[str, str]] = None, show_progress: bool = False,
                               keep_links: bool = False, batch_size: int = 50,
                               api_counters: Optional[Counter] = None) -> Dict[str, str]:
    """
    Synchronous wrapper for fetch_page_content_turbo
    
    This provides a direct drop-in replacement for functions that expect synchronous behavior.
    For async contexts, use fetch_page_content_turbo directly.
    
    Args: Same as fetch_page_content_turbo
    
    Returns:
        Dict[str, str]: A dictionary mapping URLs to their extracted content
    """
    return asyncio.run(fetch_page_content_turbo(
        urls=urls,
        use_jina=use_jina,
        jina_api_key=jina_api_key,
        snippets=snippets,
        show_progress=show_progress,
        keep_links=keep_links,
        batch_size=batch_size,
        api_counters=api_counters
    ))

# ------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"
    
    # Subscription key and endpoint for Bing Search API
    BING_SUBSCRIPTION_KEY = "YOUR_BING_SUBSCRIPTION_KEY"
    if not BING_SUBSCRIPTION_KEY:
        raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
    bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    # Perform the search
    print("Performing Bing Web Search...")
    #search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    #search_results = bing_web_search_pro(query, token="1791013312122257441", api="bing-search-pro", max_retries=10)
    search_results = google_web_search_pro(query, api_key="81b2d7ef2974da1a63669e7ffa5534a6974ff990", max_retries=10)
    print(search_results)
    
    
    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info_pro(search_results)

    print("Fetching and extracting context for each snippet...")
    
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info['url'], use_jina=False)  # Get full webpage text
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"

    print("Your Search Query:", query)
    print("Final extracted information with context:")
    print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
