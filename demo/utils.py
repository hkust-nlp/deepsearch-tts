import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import os, time
from collections import defaultdict
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List


def extract_answer_fn(output, mode='qa', extract_answer=False):
    extracted_text = ''
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
    return extracted_text


