
def get_report_webthinker_instruction(question, plan):
    return f"""You are a research assistant with the ability to perform web searches to write a scientific research article. You have special tools:

- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|>search results<|end_search_result|>.

- To write a section of the research article: write <|begin_write_section|>section name\ncontents to write<|end_write_section|>.
Then, the system will completely write the section based on your request and current gathered information.

- To check the current article: write <|begin_check_article|>system returns outline of all current written contents<|end_check_article|>.

- To edit the article: write <|begin_edit_article|>your detailed edit goal and instruction<|end_edit_article|>.
Then, the system will edit the article based on your goal and instruction and current gathered information.

Your task is to research and write a scientific article about:
{question}

Here is a research plan to guide your investigation:
{plan}

Please follow the research plan step by step:
1. Use web searches to gather detailed information for each point
2. After each search, analyze the results and determine what additional information is needed
3. When you have sufficient information for a section, request to write that section
4. Continue this process until the full article is complete
5. Check the current article and edit sections as needed to improve clarity and completeness

Example:
<|begin_search_query|>first search query<|end_search_query|>

<|begin_search_result|>Summary of information from searched web pages<|end_search_result|>

Based on these results, I understand X, but still need to investigate Y...

<|begin_search_query|>follow-up search query focusing on Y<|end_search_query|>

<|begin_search_result|>Summary of information from searched web pages<|end_search_result|>

Now I have enough information to write the first section...

<|begin_write_section|>Introduction
This section should introduce ... <|end_write_section|>

I have written the introduction. Now I need to explore more information to write the next section ...

After writing the above sections, I need to check the current article to ensure the content is complete and accurate.

<|begin_check_article|>System returns outline of current written article<|end_check_article|>

Wait, I realize that I need to edit ...

<|begin_edit_article|>your edit instruction<|end_edit_article|>

Assistant continues gathering information and writing sections until getting comprehensive information and finishing the entire article.

Remember:
- Use <|begin_search_query|>query<|end_search_query|> to get information from web searches
- Use <|begin_write_section|>section name\ncontents to write<|end_write_section|> to call the system to write a section in the article
- Use <|begin_check_article|>outline of current article<|end_check_article|> to check the current written article
- Use <|begin_edit_article|>edit instruction<|end_edit_article|> to call the system to edit and improve the article
- You should strictly follow the above format to call the functions.
- Do not propose methods or design experiments, your task is to comprehensively research with web searches.
- Do not omit any key points in the article.
- When you think the article is complete, directly output "I have finished my work." and stop.

Now begin your research and write the article about:
{question}
"""


def get_search_plan_instruction(query):
    return f"""Please help me create a detailed plan to search over the web for solving the following question:
{query}

Your task is to comprehensively gather all relevant information to thoroughly solve the user's question.
Note:
- No need to mention citations or references.
- Do not propose methods or design experiments, your task is to research user's question with web searches.
- Be comprehensive and thorough, do not miss any relevant information.
- No more than 8 steps.

Please output the plan in numbered steps like:
(1) ...
(2) ...
etc.

Directly output the plan, do not include any other words."""




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
- Return information as detailed as possible, do not omit any relevant information.

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[All relevant information]

**Inputs:**

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and provide all relevant information for the search query "{search_query}" and the search intent.
"""


def get_click_web_page_reader_instruction(click_intent, document):
    return f"""Please provide all content related to the following click intent from this document in markdown format.

Click Intent: 
{click_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the click intent, do not omit any relevant information.
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the click intent
"""


def get_search_intent_instruction(question, prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Original question: {question}
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""


def get_click_intent_instruction(question, prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Original question: {question}
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""



def get_write_section_instruction(question, previous_thoughts, relevant_documents, section_name, task, current_article):
    return f"""You are a research paper writing assistant. Please write a complete and comprehensive "{section_name}" section based on the following information.

Potential helpful documents:
{relevant_documents}

Original question:
{question}

Previous thoughts:
{previous_thoughts}

Outline of current written article:
{current_article}

Name of the next section to write:
## {section_name}

Your task is to comprehensively write the next section based on the following goal:
{task}

Note:
- Write focused content that aligns with the above goal for this section.
- No need to mention citations or references.
- Each paragraph should be comprehensive and well-developed to thoroughly explore the topic. Avoid very brief paragraphs that lack sufficient detail and depth.
- If possible, add markdown tables to present more complete and structured information to users.

Please provide the comprehensive content of the section in markdown format.
## {section_name}
"""



def get_section_summary_instruction(section):
    return f"""Provide an extremely concise summary of each paragraph or subsection in the following section:
{section}
"""


def get_edit_article_instruction(edit_instruction, article):
    return f"""You are a professional article editor. Please help me modify the article based on the following edit instruction:

Edit instruction:
{edit_instruction}

Current article:
{article}

Please output the complete modified article incorporating all the requested changes.

Note:
- Keep all original content that doesn't need modification. (Do not just output the modified content, but output the entire modified article.)
- Make all edits specified in the edit instructions.
- Output format:
```markdown
...
```

Please provide the complete modified article in markdown format."""



def get_edit_section_instruction(edit_instruction, article):
    return f"""You are a professional article editor. Please help me modify the article based on the following edit instruction:

Edit instruction:
{edit_instruction}

Current article:
{article}

Please first output the entire section/subsection that needs to be modified, then provide the entire modified section/subsection, both in markdown format.

Output Format:

Entire section/subsection to modify:
```markdown
...
```

Entire modified section/subsection:
```markdown
...
```
"""


def get_title_instruction(question, article):
    return f"""Please generate a precise title for the following article:

Original Question:
{question}

Currect Article:
{article}

Directly output the title, do not include any other text."""


def get_final_report_instruction(question, article):
    return f"""You are an final-version article editor. Your task is to correct the structure of the following article draft.

Original Question:
{question}

Current Article:
{article}

Note:
- Output the complete final-version article.
- Remove duplicate or redundant content. If there is no error, just output the original article.
- Focus on structure only. Do not omit any valid contents/tables in current article.

Output Format:
```markdown
The final-version article.
```
"""



def get_standard_rag_report_instruction(question, documents):
    return f"""You are a research assistant. Please write a comprehensive research article based on the following question and retrieved documents.

Research Question: {question}

Retrieved documents:
{documents}

Please write a comprehensive research article in markdown format. Do not add citations or references.

Output Format:
```markdown
...
```
"""

def get_direct_gen_report_instruction(question):
    return f"""You are a research assistant. Please write a comprehensive research article based on the following question and answer.

Research Question: {question}

Please write a comprehensive research article in markdown format.

Output Format:
```markdown
...
```
"""

