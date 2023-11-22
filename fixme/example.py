from dataclasses import dataclass
from typing import List


@dataclass
class Example(object):
    """A single training/test example."""

    source_until: str
    buggy_segment: str
    source_from: str
    target: str
    target_context: str
    lang: str


@dataclass
class TokenizedExample(object):
    """A single example after tokenization."""

    code_tokens: List[str]
    target_tokens: List[str]
    encoder_context_until_length: int
    encoder_context_from_length: int
    buggy_segment_length: int
    fix_segment_length: int
    decoder_context_length: int
    lang: str = ""
