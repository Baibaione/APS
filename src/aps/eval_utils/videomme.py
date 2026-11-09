# Adopted from lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval).
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an AS IS basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Local experiment adaptations are documented in third_party/NOTICE.md.
import re
import random
import numpy as np

def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = 'Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.'
    question = doc['question']
    option = '\n'.join([f'{opt}' for i, opt in enumerate(doc['options'])])
    question = question + '\n' + option
    post_prompt = lmms_eval_specific_kwargs['post_prompt'] if 'post_prompt' in lmms_eval_specific_kwargs else 'The best answer is:'
    full_prompt = option_prompt + '\n' + question + '\n' + post_prompt
    return full_prompt

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = ['The best answer is', 'The correct answer is', 'The answer is', 'The answer', 'The best option isThe correct option is', 'Best answer:Best option:']
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')
    if len(s.split()) > 10 and (not re.search('[ABCD]', s)):
        return ''
    matches = re.search('[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]
