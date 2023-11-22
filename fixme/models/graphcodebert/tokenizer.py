from typing import Tuple
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from .parser import extract_dataflow, parsers, mlang2dfglang
from ...example import Example, TokenizedExample
from .dataset import DFG, ORI2CUR


def merge_dfgs(before: DFG, after: DFG, offset: int) -> DFG:
    merged = []
    name2offset = {}
    for node in before:
        varname = node[0]
        token_offset = node[1]
        name2offset[varname] = token_offset
        merged.append(node)

    for node in after:
        varname = node[0]
        token_offset = node[1] + offset
        reference_names = node[3]
        reference_offsets = [reference + offset for reference in node[4]]

        # if var appears for the first time add reference to appearence in before
        if varname in name2offset and len(reference_names) == 0:
            reference_names.append(varname)
            reference_offsets.append(name2offset[varname])

        node = (varname, token_offset, "comesFrom", reference_names, reference_offsets)
        merged.append(node)
    return merged


def tokenize_example(
    example: Example, tokenizer: RobertaTokenizer, idx: int
) -> Tuple[TokenizedExample, DFG, ORI2CUR]:
    lang = mlang2dfglang[example.lang]
    tokens_before, dfg_before = extract_dataflow(
        example.source_until,
        parsers[lang],
        lang,
    )
    tokens_before = [
        tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(tokens_before)
    ]
    tokens_segment, dfg_segment = extract_dataflow(
        example.buggy_segment,
        parsers[lang],
        lang,
    )
    tokens_segment = [tokenizer.tokenize(x) for x in tokens_segment]
    tokens_after, dfg_after = extract_dataflow(
        example.source_from,
        parsers[lang],
        lang,
    )
    tokens_after = [tokenizer.tokenize(x) for x in tokens_after]
    code_tokens = (
        [[tokenizer.cls_token]]
        + tokens_before
        + [[tokenizer.sep_token]]
        + tokens_segment
        + [[tokenizer.sep_token]]
        + tokens_after
        + [[tokenizer.sep_token]]
    )
    dfg = merge_dfgs(
        merge_dfgs(
            dfg_before,
            dfg_segment,
            len(tokens_before) + 1,
        ),
        dfg_after,
        len(tokens_before) + len(tokens_segment) + 2,
    )

    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i, token in enumerate(code_tokens):
        ori2cur_pos[i] = (
            ori2cur_pos[i - 1][1],
            ori2cur_pos[i - 1][1] + len(token),
        )

    code_tokens = [y for x in code_tokens for y in x]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

    target_context_tokens = tokenizer.tokenize(example.target_context)
    target_fix_tokens = tokenizer.tokenize(example.target)
    fix_token_length = len(target_fix_tokens) + 1
    decoder_context_length = len(target_context_tokens) + 2
    target_tokens = target_context_tokens + [tokenizer.sep_token] + target_fix_tokens
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    code_before_token_length = len([y for x in tokens_before for y in x])
    code_segment_token_length = len([y for x in tokens_segment for y in x])
    code_after_token_length = len([y for x in tokens_after for y in x])

    return (
        TokenizedExample(
            code_tokens=code_tokens,
            target_tokens=target_tokens,
            encoder_context_until_length=code_before_token_length,
            encoder_context_from_length=code_after_token_length,
            buggy_segment_length=code_segment_token_length,
            fix_segment_length=fix_token_length,
            decoder_context_length=decoder_context_length,
        ),
        dfg,
        ori2cur_pos,
    )
