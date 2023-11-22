from typing import Callable
from .example import TokenizedExample

CutterFn = Callable[[TokenizedExample, int, int], TokenizedExample]


def cut_context(
    sample: TokenizedExample,
    max_source_length: int,
    window_length: int,
) -> TokenizedExample:
    """Cuts the sample by limiting cutting the encoder and decoder context as little as possible."""

    start = 0
    if len(sample.code_tokens) > max_source_length:
        if sample.buggy_segment_length > max_source_length:
            start = max(sample.encoder_context_until_length + 2 - window_length, 0)
        else:
            max_source_length_before = window_length
            max_source_length_after = max(
                max_source_length - sample.buggy_segment_length - window_length,
                0,
            )

            if sample.encoder_context_from_length <= max_source_length_after:
                start = max(len(sample.code_tokens) - max_source_length, 0)
            elif sample.encoder_context_until_length > max_source_length_before:
                start = max(
                    sample.encoder_context_until_length - max_source_length_before + 1,
                    0,
                )
    sample.code_tokens = sample.code_tokens[start:]
    target_length = sample.fix_segment_length + window_length
    sample.target_tokens = sample.target_tokens[-target_length:]
    return sample
