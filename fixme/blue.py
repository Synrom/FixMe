# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


smoothie = SmoothingFunction().method4

def compute_blue(references: List[List[str]], candidates: List[List[str]]):
    bleu_scores = [sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie) for reference, candidate in zip(references, candidates)]
    return np.mean(bleu_scores)

def compute_blue_file(ref_file: str, trans_file: str) -> float:
    reference_text: List[str] = []
    with open(ref_file) as fh:
        reference_text = fh.readlines()
    per_segment_references: List[List[str]] = []
    for reference in reference_text:
        per_segment_references.append(reference.strip().split())
    translations: List[List[str]] = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())
    bleu_score = compute_blue(
        per_segment_references, translations
    )
    return round(100 * bleu_score, 2)
