# Adopted from lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval).
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an AS IS basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Local experiment adaptations are documented in third_party/NOTICE.md.
import re
import random
import numpy as np

def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    question = doc['question']
    pre_prompt = lmms_eval_specific_kwargs.get('pre_prompt', '')
    post_prompt = lmms_eval_specific_kwargs.get('post_prompt', '')
    return pre_prompt + question + post_prompt

def extract_characters_regex(s):
    """Port of lmms_eval.tasks._task_utils.mcq_extract.extract_mcq_answer(s, ['A'..'D'])."""
    choices = ['A', 'B', 'C', 'D']
    answer_phrases = ['the answer is', 'answer is', 'the correct answer is', 'correct answer is', 'the best answer is', 'best answer is', 'the correct option is', 'correct option is', 'the best option is', 'best option is', 'the choice is', 'choice is', 'the correct choice is', 'correct choice is', 'i choose', 'i select', 'i pick', 'my answer is', 'my choice is']
    format_priority = {'start': 10, 'end': 9, 'phrase': 7, 'parentheses': 6, 'period': 5, 'colon': 4, 'right_paren': 3, 'space': 2, 'fallback': 0}
    if not s or not s.strip():
        return ''
    text = s.strip()
    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        text = text.strip(char)
    text = ' ' + text + ' '
    candidates = []
    for ch in choices:
        if f'({ch})' in text:
            candidates.append((ch, text.rfind(f'({ch})'), 'parentheses'))
    for ch in choices:
        if f'{ch}.' in text:
            candidates.append((ch, text.rfind(f'{ch}.'), 'period'))
    for ch in choices:
        if f'{ch}:' in text:
            candidates.append((ch, text.rfind(f'{ch}:'), 'colon'))
    for ch in choices:
        if f'{ch})' in text:
            candidates.append((ch, text.rfind(f'{ch})'), 'right_paren'))
    for ch in choices:
        if f'{ch} ' in text:
            candidates.append((ch, text.rfind(f'{ch} '), 'space'))
    text_lower = text.lower()
    for phrase in answer_phrases:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for ch in choices:
                ch_pos = text.find(ch, after)
                if ch_pos != -1:
                    candidates.append((ch, ch_pos, 'phrase'))
    stripped = text.strip()
    for ch in choices:
        if stripped.startswith(ch) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((ch, 0, 'start'))
    for ch in choices:
        if stripped.endswith(ch) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((ch, len(text) - 1, 'end'))
    if not candidates:
        for ch in choices:
            if ch in text:
                candidates.append((ch, text.rfind(ch), 'fallback'))
    if not candidates:
        return ''
    candidates.sort(key=lambda x: (format_priority.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]
