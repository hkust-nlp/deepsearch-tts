from typing import List, Dict, Any
import requests
import json
import random
import os
import logging

logger = logging.getLogger(__name__)

def search_google(query_list: list, num_results: int = 10) -> List[Dict[str, Any]]:
        '''Use Google search engine to search information for the given query. Google is usually a good choice. Translate your query into English for better results unless the query is Chinese localized.

        Args:
            query_list (list): The list of queries to be searched(List[str]). Search a list of queries at a time is highly recommended. Each query should be distinctive and specific. e.g. ['xx xxxx xx', 'xxxx', ...].
            num_results (int): The number of result pages to retrieve for EACH query. default is 4.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a website.
                Each dictionary contains the following keys:
                - 'title': The title of the website.
                - 'link': The URL of the website.
                - 'snippet': A brief description of the website.
                - 'position': A number in order.
                - 'sitelinks': Useful links within the website.

                Example:
                {
                    'title': 'OpenAI',
                    'link': 'https://www.openai.com'
                    'snippet': 'An organization focused on ensuring that
                    'position': 1,
                    'sitelinks': [...],
                }
            title, description, url of a website.
        '''
        if isinstance(query_list, str):
            query_list = [query_list]
        all_results=[]
        for query in query_list:
            # google_search_access = get_google_serper_access()
            # igoogle = random.choice(range(len(google_search_access)))
            GOOGLE_API_KEY = "81b2d7ef2974da1a63669e7ffa5534a6974ff990"
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': GOOGLE_API_KEY,
                'Content-Type': 'application/json'
            }
            payload = json.dumps({
                "q": query,
                "num": num_results
            })
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                responses = json.loads(response.text)['organic']
            except Exception as e:
                # if len(google_search_access)>=2:
                #     del_google = google_search_access.pop(igoogle)
                #     logger.error(f'DELETE GOOGLE SERPER ACCESS KEY: {del_google}')
                # else:
                #     logger.error(f'DEATH ERROR IN GOOGLE SERPER SEARCH: {repr(e)}. NO AVAILABLE KEY LEFT.')
                #     os._exit(1)
                logger.error(f"Google search failed with error: {repr(e)}")
                responses=[{"error": f"google serper search failed for {query=}. The error is: {repr(e)}"}]
            
            all_results.extend(responses)
        
        return all_results
    
    
    
print(search_google(["Who is Weihao Zeng?"]))