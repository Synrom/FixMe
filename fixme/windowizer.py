from typing import Callable, List
from .example import TokenizedExample

WindowizerFn = Callable[[TokenizedExample, int, int, int], List[TokenizedExample]]


def windowize(
    sample: TokenizedExample,
    max_source_length: int,
    max_target_length: int,
    sep_token: int,
) -> List[TokenizedExample]:
    window_length = int(max_target_length / 2)
    start = 0
    source_limit = max(len(sample.code_tokens) - max_source_length, 0)
    windows: List[TokenizedExample] = []
    finished = False
    window_offset = 0
    while window_offset < sample.buggy_segment_length or window_offset < sample.fix_segment_length:
        #for window_offset in range(0, sample.buggy_segment_length, window_length):
        # source
        source_tokens = sample.code_tokens[start:]
        source_tokens = source_tokens[:max_source_length]

        if finished:
            break
        predict_window_start = -sample.fix_segment_length + window_offset
        if predict_window_start + window_length >= 0:
            finished = True

        predict_window = sample.target_tokens[
            predict_window_start:
        ][:window_length]
        target_context_window = sample.target_tokens[
            : predict_window_start
        ][-window_length:]
        target_window = target_context_window + predict_window

        predict_token_length = len(predict_window)
        target_token_length = len(target_window)
        if window_offset == 0 and predict_window[0] == sep_token:
            predict_token_length -= 1

        encoder_context_until_length = max(
            sample.encoder_context_until_length - start, 0
        )
        encoder_context_from_length = max(
            start
            + max_source_length
            - (sample.encoder_context_until_length + sample.buggy_segment_length),
            0,
        )
        buggy_segment_length = (
            max_source_length
            - encoder_context_from_length
            - encoder_context_until_length
        )

        windows.append(
            TokenizedExample(
                code_tokens=source_tokens,
                target_tokens=target_window,
                encoder_context_until_length=encoder_context_until_length,
                encoder_context_from_length=encoder_context_from_length,
                buggy_segment_length=buggy_segment_length,
                fix_segment_length=predict_token_length,
                decoder_context_length=target_token_length - predict_token_length,
                lang=sample.lang,
            )
        )

        start = min(start + window_length, source_limit)
        window_offset += window_length
    return windows


def no_windowization(
    sample: TokenizedExample, max_source_length: int, max_target_length: int, _: int
) -> List[TokenizedExample]:
    if sample.buggy_segment_length > max_source_length:
        sample.code_tokens=sample.code_tokens[
            sample.encoder_context_until_length : sample.encoder_context_until_length
            + max_source_length
        ]
    sample.code_tokens = sample.code_tokens[:max_source_length]
    sample.target_tokens = sample.target_tokens[-max_target_length:]
    sample.fix_segment_length = min(sample.fix_segment_length, max_target_length)
    sample.encoder_context_from_length = 0 # isnt used anyways
    sample.encoder_context_until_length = 0 # isnt used anyways
    sample.decoder_context_length=max(max_target_length - sample.fix_segment_length,0),
    sample.buggy_segment_length=min(max_source_length, sample.buggy_segment_length),
    return [sample]
