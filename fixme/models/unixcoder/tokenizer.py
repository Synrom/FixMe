from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from ...example import Example, TokenizedExample


def tokenize_example(example: Example, tokenizer: RobertaTokenizer) -> TokenizedExample:
    sul = len(example.source_until)
    sfl = len(example.source_from)
    bsl = len(example.buggy_segment)
    dcl = len(example.target_context)
    tl = len(example.target)
    tokens_before = tokenizer.tokenize(example.source_until)
    tokens_segment = tokenizer.tokenize(example.buggy_segment)
    tokens_after = tokenizer.tokenize(example.source_from)

    code_tokens = (
        tokens_before
        + [tokenizer.sep_token]
        + tokens_segment
        + [tokenizer.sep_token]
        + tokens_after
    )

    target_context_tokens = tokenizer.tokenize(example.target_context)
    target_fix_tokens = tokenizer.tokenize(example.target)
    fix_token_length = len(target_fix_tokens) + 1
    decoder_context_length = len(target_context_tokens) + 2
    target_tokens = target_context_tokens + [tokenizer.sep_token] + target_fix_tokens
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]

    return TokenizedExample(
        code_tokens=code_tokens,
        target_tokens=target_tokens,
        encoder_context_until_length=len(tokens_before),
        encoder_context_from_length=len(tokens_after),
        buggy_segment_length=len(tokens_segment),
        fix_segment_length=fix_token_length,
        decoder_context_length=decoder_context_length,
        lang=example.lang,
    )
