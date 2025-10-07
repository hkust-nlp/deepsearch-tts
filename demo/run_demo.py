import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
import streamlit as st
import asyncio
import time
import json_repair
import re
from run_logit import process_query_async
from settings import Environment

@st.cache_resource
def init_env():
    print("Initializing environment...")
    if 'env_initialized' not in st.session_state:
        env = Environment()
        st.session_state.env = env
        st.session_state.env_initialized = True
        print("Environment initialization completed")
    else:
        env = st.session_state.env
        print("Using existing environment")
    
    return env

async def summarize_thought_chain(env, reasoning_chain):
    client = env.aux_client
    instruction = f'''Please analyze the given model thought chain segment and complete two tasks:
    1. Generate a concise title (title) summarizing the current operation in the thought chain. You can add an appropriate emoji icon at the beginning of the title to represent the current action. Use common emojis.
    2. Write a first-person explanation (explain) describing what the thought chain is doing, what problems were encountered, or what the next steps are. If the thought chain mentions specific webpage information or factual information, please include it in the explanation.

    Please provide the output in the following JSON format:
    {{"title": "title here", "explain": "explanation here"}}

    Example:
    {{"title": "🔍 Information Gap Found", "explain": "While the website provided insights about the school's vision, I haven't found specific details about its history and mission. This is an area I need to investigate further to provide a comprehensive overview."}}

    Please ensure the output JSON contains both title and explain.

    Thought chain:
    {reasoning_chain}
    '''
    prompt = instruction
    prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'

    response = await client.completions.create(
        model=env.aux_model_name,
        max_tokens=4096,
        prompt=prompt,
        timeout=3600,
    )
    response = response.choices[0].text
    response = json_repair.loads(response)
    if isinstance(response,list):
        response = response[0]
    if not isinstance(response, dict):
        print("Error in summary title")
        return '', ''
    title = response.get('title','')
    explain = response.get('explain','')

    title = title.replace('，',', ').replace('。','. ')
    explain = explain.replace('，',', ').replace('。','. ')
    return title, explain

async def app():
    st.set_page_config(
        page_title="WebThinker",
        layout="centered" 
    )
    
    # 设置页面样式
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 800px;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .title {
        text-align: center;
        margin-bottom: 2rem;
        width: 100%;
    }

    .stTextInput, 
    .element-container:has(.thinking-completed),
    .element-container:has(.answer-section),
    .stMarkdown:has(> div) > div:first-child,
    .stMarkdown:has(> div) > div > div {  
        width: 100% !important;
        max-width: 800px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    div.stTextInput > div > div > input {
        width: 100% !important;
    }

    .thinking-completed, 
    .answer-section {
        width: 100% !important;  
        padding: 20px !important;
        margin: 1rem 0 !important;
        box-sizing: border-box !important;  
    }

    .thinking-completed {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .answer-section {
        border: 1px solid #4CAF50;
        border-radius: 5px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stMarkdown {
        width: 100% !important;
        max-width: 100% !important;
    }

    .stMarkdown > div > div {
        width: 100% !important;
        max-width: 100% !important;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .thinking-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 0, 0, 0.1);
        border-radius: 50%;
        border-top-color: #4CAF50;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    .thinking-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="title"><h1>WebThinker</h1></div>', unsafe_allow_html=True)
        query = st.text_input("Enter your question：", "", key="query_input")

    if query:
        print(f"Processing query: {query}")
        if 'env' not in st.session_state or 'env_initialized' not in st.session_state:
            env = init_env()
            st.session_state.env = env
        else:
            env = st.session_state.env
            env.reset()
            
        st.sidebar.title("Thoughts")
        
        with st.container():
            thinking_container = st.empty()  
            answer_container = st.empty()    
        
        sidebar_container = st.sidebar.empty()  
        
        thinking_process = "" 
        current_chain = ""     
        summarized_process = "" 
        final_answer = ""
        answer_started = False
        newline_count = 0    
        
        thinking_status = st.empty()
        
        try:
            thinking_status.markdown('''
                <div class="thinking-header">
                    <div class="thinking-spinner"></div>
                    <span>Thinking in progress...</span>
                </div>
            ''', unsafe_allow_html=True)
            
            summary_tasks = []
            
            async for chunk in process_query_async(query, st.session_state.env):
                if chunk:
                    if not answer_started:
                        thinking_process += chunk
                        current_chain += chunk

                        if '\\boxed{' in thinking_process:
                            answer_started = True
                            final_answer = thinking_process.split('\\boxed{')[-1]
                            thinking_process = thinking_process.split('\\boxed{')[0]
                            current_chain = current_chain.split('\\boxed{')[0]

                            if current_chain.strip():
                                summary_tasks.append(asyncio.create_task(
                                    summarize_thought_chain(st.session_state.env, current_chain)
                                ))

                            thinking_container.markdown(f'<div class="thinking-completed">{summarized_process}</div>', unsafe_allow_html=True)
                            answer_container.markdown(f'<div class="answer-section"><h3>🎯 Final Answer：</h3>{final_answer}</div>', unsafe_allow_html=True)

                        else:
                            newline_count = current_chain.count('\n\n')
                            if newline_count >= 3:
                                if current_chain.strip():
                                    summary_tasks.append(asyncio.create_task(
                                        summarize_thought_chain(st.session_state.env, current_chain)
                                    ))
                                
                                current_chain = ""
                                newline_count = 0

                    else:
                        thinking_process += chunk
                        final_answer += chunk
                        thinking_container.markdown(f'<div class="thinking-completed">{summarized_process}</div>', unsafe_allow_html=True)
                        answer_container.markdown(f'<div class="answer-section"><h3>🎯 Final Answer：</h3>{final_answer}</div>', unsafe_allow_html=True)

                    search_pattern = r'<\|begin_search_query\|>.*?<\|end_search_query\|>'
                    click_pattern = r'<\|begin_click_link\|>.*?<\|end_click_link\|>'
                    thinking_process = re.sub(search_pattern, '', thinking_process, flags=re.DOTALL)
                    thinking_process = re.sub(click_pattern, '', thinking_process, flags=re.DOTALL)
                    thinking_process = thinking_process.replace('Final Information','')
                    sidebar_container.markdown(thinking_process)
                    
                    done_tasks = []
                    for task in summary_tasks:
                        if task.done():
                            title, summary = await task
                            summarized_process += f"#### {title}\n{summary}\n\n"
                            done_tasks.append(task)
                            thinking_container.markdown(summarized_process)
                    
                    for task in done_tasks:
                        summary_tasks.remove(task)
                        
                    await asyncio.sleep(0.05)
            
            if summary_tasks:
                for task in asyncio.as_completed(summary_tasks):
                    title, summary = await task
                    summarized_process += f"### {title}\n{summary}\n\n"
                    thinking_container.markdown(summarized_process)
            final_answer = final_answer.strip().rstrip("}")
            if thinking_process or final_answer:
                sidebar_container.markdown(thinking_process + '\n\n---\n\nFinished!')
                thinking_container.markdown(summarized_process)
                if final_answer:
                    answer_container.markdown(f'<div class="answer-section"><h3>🎯 Final Answer：</h3>{final_answer}</div>', unsafe_allow_html=True)
            
            thinking_status.empty()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    asyncio.run(app())