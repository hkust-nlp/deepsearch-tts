
def get_gpqa_web_thinker_instruction(MAX_SEARCH_LIMIT=15):
    return """You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:

- To perform a search: write <|begin_search_query|>your query here<|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. Once you have all the information you need, continue your reasoning.

Example:
Question: "What is the energy range of pp III neutrinos?"
Thinking steps:
- I might need to look up details about pp III neutrinos.

<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>

(System returns processed information from relevant web pages)

Continues reasoning with the new information...

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.
"""





def get_deep_web_explorer_instruction(search_query, search_intent, search_result):
    return f"""You are a web explorer analyzing search results to find relevant information based on a given search query and search intent.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **More Information Seeking:**
- If the information is not relevant to the query, you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the **Current Search Query**.

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and extract relevant information for the search query "{search_query}" and the search intent.
"""


def get_deep_web_explorer_instruction_qwen(search_intent, search_result):
    return f"""You are a web explorer analyzing search results to find relevant information based on the goal of the visit for webpage(s).

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the goal of the visit for webpage(s).

2. **More Information Seeking:**
- If the information is not relevant to the goal of the visit for webpage(s), you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the goal of the visit for webpage(s).

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Detailed Goal of the Visit for Webpage(s):**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and extract relevant information for the goal of the visit for webpage(s).
"""


def get_web_page_reader_instruction(query, document):
    return f"""{document}
Please provide all content related to "{query}" from this document in markdown format.
If there isn't any relevant information, just output "No relevant information". If there is any relevant information, output all the relevant information with potential helpful links."""

def get_detailed_web_page_reader_instruction(query, search_intent, document):
    return f"""Please provide all content related to the following search query and search intent from this document in markdown format.

Search Query: 
{query}

Search Intent: 
{search_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the search query and intent, do not omit any relevant information.
- Include any relevant links from the source
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the query/intent
"""


def get_search_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""


def get_click_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""



def get_query_plan_instruction(question):
    return f"""You are a reasoning assistant. Your task is to generate a detailed query plan for answering the user's question by breaking it down into sub-queries.

Question: {question}

Please analyze the question and break it down into multiple sub-queries that will help gather all the necessary information to answer it completely. 

Output your query plan in JSON format as follows:

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
```
"""









def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_timeline_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you create an accurate chronological timeline summary. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        "You should perform multiple searches to gather comprehensive information until you believe you have enough details.\n"
        "Finally, provide a comprehensive timeline that includes all relevant events in chronological order.\n\n"
        "Example:\n"
        "Text: \"Create a timeline of key events in the Apollo 11 mission.\"\n"
        "Assistant thinking steps:\n"
        "- I need to find key dates and events of the Apollo 11 mission.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Apollo 11 mission timeline key events dates<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- You should perform as many searches as possible to gather comprehensive information.\n\n"
    )



def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )
    
def get_web_verifier_instruction(question, predicted_answer):
    return f"""You are a reasoning assistant with the ability to perform web searches to help you verify whether the predicted answer is correct for the user's question. You have special tools:

- To perform a search: write <|begin_search_query|>your query here<|end_search_query|>.
    
    The system will then search and analyze relevant web pages and provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.
    

You can repeat the search process multiple times if necessary. Once you have all the information you need, continue your reasoning.

Remember:

- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.

Please verify the following predicted answer, and provide a confidence score between 0 and 1 indicating the probability that the predicted answer is correct. Think step by step as you verify it.

Provide your final confidence in the format \\boxed{{}}. 

Question: 
{question}
Predicted Answer: 
{predicted_answer}

"""


def get_web_verifier_instructionv1(question, predicted_answer):
    return f"""You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** whether the predicted answer to the user's question is accurate. To reduce the risk of hallucinations and improve reliability, **you are encouraged to actively use web searches whenever needed**.

You have special tools to assist in this process:

- To perform a search, use:
  <|begin_search_query|> your query here <|end_search_query|>

- The system will return results in the format:
  <|begin_search_result|> ...search results... <|end_search_result|>

You may search multiple times if necessary to gather enough evidence.

**Please follow this workflow:**
1. Read the user's question and the predicted answer.
2. If the answer contains any uncertain, factual, or verifiable elements, perform a web search.
3. Analyze the returned search results to determine if the answer is correct.
4. Continue reasoning using the verified information.

Your goal is to provide a **final confidence score between 0 and 1**, reflecting how likely the predicted answer is to be correct based on the information you gathered.

Use the format:
\\boxed{{<confidence_score>}}

Question: 
{question}
Predicted Answer: 
{predicted_answer}
"""


def get_web_verifier_instructionv2(question, predicted_answer):
    return f"""You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** whether the predicted answer to the user's question is accurate. To reduce the risk of hallucinations and improve reliability, **you are encouraged to actively use web searches whenever needed**.

You have special tools to assist in this process:

- To perform a search, use:
  <|begin_search_query|> your query here <|end_search_query|>

- The system will return results in the format:
  <|begin_search_result|> ...search results... <|end_search_result|>

You may search multiple times if necessary to gather enough evidence.

**Verification Process:**
1. Carefully analyze the user's question and identify all explicit or implicit conditions that a correct answer must satisfy.
2. List these required conditions clearly.
3. Examine whether the predicted answer satisfies **each condition**.
4. Use web searches to gather evidence and validate uncertain parts of the predicted answer.
5. Based on how many of the required conditions are satisfied by the predicted answer, assign a confidence score between 0 and 1:
   - If all conditions are fully satisfied → confidence = 1.
   - If only some conditions are satisfied → confidence = (number of satisfied conditions / total conditions).
   - If none are satisfied → confidence = 0.

Your goal is to give a reasoned, evidence-backed assessment.

Return your final confidence score using the format:
\\boxed{{<confidence_score>}}

Question: 
{question}
Predicted Answer: 
{predicted_answer}
"""

def get_web_verifier_instructionv3(question, predicted_answer):
    return f"""You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** whether the predicted answer to the user's question is accurate. Your goal is to avoid hallucinations and base your judgment on verifiable, external information.

To ensure reliability, you are **strongly encouraged to perform web searches to check the factual accuracy of any part of the predicted answer**.

You have special tools:

- To perform a search, use:
  <|begin_search_query|> your query here <|end_search_query|>

- Search results will be returned as:
  <|begin_search_result|> ...search results... <|end_search_result|>

You may perform multiple searches if necessary.

**Verification Procedure:**

1. **Carefully read the question** and identify **all explicit and implicit conditions** that a correct answer must satisfy.
2. **List each of these conditions** clearly.
3. For each condition, **check if the predicted answer satisfies it**, using **web searches to support your verification**.
4. Count how many conditions are satisfied.
5. Assign a confidence score between 0 and 1 based on the fraction of conditions that are met:
   - If **all conditions** are fully satisfied → confidence = 1.
   - If **some conditions** are satisfied → confidence = (number of satisfied conditions / total number of conditions).
   - If **none are satisfied** → confidence = 0.

You must justify your reasoning based on evidence from the search.

Return your final confidence score using the format:
\\boxed{{<confidence_score>}}

Question:  
{question}

Predicted Answer:  
{predicted_answer}
"""


def get_web_verifier_instructionv4(question, predicted_answer):
    return f"""You are a reasoning assistant equipped with web search capabilities to **fact-check and verify** whether the predicted answer to the user's question is accurate. To reduce the risk of hallucinations and improve reliability, **you are encouraged to actively use web searches whenever needed**.

You have special tools to assist in this process:

- To perform a search, use:
  <|begin_search_query|> your query here <|end_search_query|>

- The system will return results in the format:
  <|begin_search_result|> ...search results... <|end_search_result|>

You may search multiple times if necessary to gather enough evidence.

**Verification Process:**
1. Carefully analyze the user's question and identify all explicit or implicit conditions that a correct answer must satisfy.
2. List these required conditions clearly.
3. For **each condition**, use web searches to verify if the predicted answer satisfies it, providing explicit evidence from the search results.
4. Based on how many conditions have clear search-backed evidence, assign a confidence score between 0 and 1:
   - If all conditions have explicit web evidence → confidence = 1.
   - If only some conditions have explicit web evidence → confidence = (number of conditions with evidence / total conditions).
   - If none have explicit web evidence → confidence = 0.

Your goal is to give a reasoned, evidence-backed assessment.

Return your final confidence score using the format:
\\boxed{{<confidence_score>}}

Question: 
{question}
Predicted Answer: 
{predicted_answer}
"""


def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following question.\n\n'
            'Provide your final answer in the format **ANSWER: {YOUR_ANSWER}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following math question.\n\n'
            'Provide your final answer in the format **ANSWER: YOUR_ANSWER**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following multiple-choice question.\n\n'
            'Provide your final choice in the format **ANSWER: {YOUR_CHOICE}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt

def get_task_instruction_timeline(text, model_name=None):
    # Common format template for both cases
    format_template = '- [DATE/TIME]: Event description\n\n'
    # Base prompt that's shared between both cases
    base_prompt = f'Text:\n{text}\n\n'
    if model_name == 'qwq':
        return (
            'Now it is March 14, 2025. Please create a comprehensive timeline based on the given text.'
            f'Format each event as:\n{format_template}'
            'Ensure events are ordered chronologically and include specific dates/times when available.\n\n'
            f'{base_prompt}'
        )
    else:
        return (
            'Please summarize the key events from the text in chronological order. '
            'For each event, include the date/time (if available) and a clear description.\n\n'
            f'Format your timeline as:\n{format_template}'
            f'{base_prompt}'
        )

