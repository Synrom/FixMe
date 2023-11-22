from typing import Callable, Dict
from .example import Example

PreCutterFn = Callable[[Example, int, int, int], Example]

def cut_too_long(
    example: Example,
    max_source_length: int,
    max_target_length: int,
    max_token_length: int,
    long_tokens: Dict[str, int] = {}
) -> Example:
    """Cuts the sample by limiting cutting the encoder and decoder context as little as possible."""

    for string in sorted(long_tokens, key=lambda k: long_tokens[k], reverse=True):
        if string in example.source_until:
            max_target_length = long_tokens[string]
            break
        if string in example.source_from:
            max_target_length = long_tokens[string]
            break

    max_source_length = max_source_length * max_token_length
    max_target_length = max_target_length * max_token_length

    example.source_until = example.source_until[-max_source_length:]
    example.source_from = example.source_from[:max_source_length]    
    example.target_context = example.target_context[-max_target_length:]

    return example

def no_cutter(
    example: Example,
    max_source_length: int,
    max_target_length: int,
    max_token_length: int,
    long_tokens: Dict[str, int] = {}
) -> Example:
    return example

