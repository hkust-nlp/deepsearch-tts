import time
import requests
from openai import AsyncOpenAI


class Environment:
    def __init__(
            self, 
            use_model_name='QwQ-32B',
            aux_model_name='Qwen2.5-72B-Instruct',
            max_search_limit=15,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            top_k=20,
            min_p=0.05,
            search_num=10,
            max_interation_times=10,
            max_path_tokens=20000,
            api_base_url="",
            aux_api_base_url='',
            bing_subscription_key="",
            bing_endpoint="https://api.bing.microsoft.com/v7.0/search",
            lora_name=None,
            lora_path=None,
            use_jina=False,
            jina_api_key=None,
            keep_links=True,
        ):
        
        self.use_model_name = use_model_name
        self.aux_model_name = aux_model_name
        self.max_search_limit = max_search_limit
        self.jina_api_key = jina_api_key
        self.use_jina = use_jina
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.min_p = min_p
        self.search_num = search_num
        self.max_path_tokens = max_path_tokens
        self.max_interation_times = max_interation_times
        self.start_time = time.time()
        self.bing_subscription_key = bing_subscription_key
        self.bing_endpoint = bing_endpoint
        self.keep_links = keep_links
        self.search_cache = {}
        self.url_cache = {}
        self.api_base_url = api_base_url
        self.aux_api_base_url = aux_api_base_url
        self.lora_name = lora_name
        self.lora_path = lora_path

        self.error_indicators = [
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
        
        self._load_all()

    def _load_all(self):
        self._load_special_tokens()
        self._load_client(self.api_base_url, self.aux_api_base_url)
        self._load_lora(self.lora_name, self.lora_path)
        self._load_init_vars()
        
    def _load_special_tokens(self):
        self.BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
        self.END_SEARCH_QUERY = "<|end_search_query|>"
        self.BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
        self.END_SEARCH_RESULT = "<|end_search_result|>"
        self.BEGIN_CLICK_LINK = "<|begin_click_link|>"
        self.END_CLICK_LINK = "<|end_click_link|>"
        self.BEGIN_CLICK_RESULT = "<|begin_click_result|>"
        self.END_CLICK_RESULT = "<|end_click_result|>"
    def _load_client(self, api_base_url, aux_api_base_url):
        self.client = AsyncOpenAI(
            api_key="empty",
            base_url=api_base_url,
        )
        self.aux_client = AsyncOpenAI(
            api_key="empty",
            base_url=aux_api_base_url,
        )
    
    def _load_lora(self, lora_name, lora_path):
        if lora_name is None or lora_path is None:
            return
        try:
            lora_load_url = f"{self.api_base_url}/load_lora_adapter"
            lora_payload = {
                "lora_name": lora_name,
                "lora_path": lora_path
            }
            requests.post(lora_load_url, json=lora_payload)
            return True
        except Exception as e:
            print(f"Error loading LoRA adapter: {e}")
            return False

    def _load_init_vars(self):
        self.search_count = 0
        self.interation_times = 0
        self.total_tokens = 0
        self.executed_search_queries = set()
        self.clicked_urls = set()
        self.prompt = None
        self.total_tokens = 0
        self.output = ''
        self.history = []
    
    def reset(self):
        self._load_init_vars()
    
    def update_step(self, step):
        self.history.append(step)
        self.prompt += step
        self.total_tokens += len(step.split())
        self.output += step
        self.interation_times += 1

    def update_search(self, search_query):
        self.search_count += 1
        self.interation_times += 1
        self.executed_search_queries.add(search_query)
    
    def update_click(self, url):
        self.clicked_urls.add(url)
        self.interation_times += 1
    def add_child_env(self):
        child_env = SubEnvironment(
            use_model_name=self.use_model_name,
            aux_model_name=self.aux_model_name,
            max_search_limit=self.max_search_limit,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            min_p=self.min_p,
            search_num=self.search_num,
            max_interation_times=self.max_interation_times,
            max_path_tokens=self.max_path_tokens,
            api_base_url=self.api_base_url,
            aux_api_base_url=self.aux_api_base_url,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            use_jina=self.use_jina,
            jina_api_key=self.jina_api_key,
            keep_links=self.keep_links,
        )
        self.history.append(child_env)
        child_env.search_cache = self.search_cache
        child_env.url_cache = self.url_cache
        return child_env
    

class SubEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_all(self):
        self._load_special_tokens()
        self._load_init_vars()
        
        
        
        
