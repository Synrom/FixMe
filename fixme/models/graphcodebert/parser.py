import os
from tree_sitter import Language, Parser
from typing import Dict, List
from .parser_helpers import (
    DFG_python,
    DFG_java,
    DFG_ruby,
    DFG_go,
    DFG_php,
    DFG_javascript,
    DFG_csharp,
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
)

mlang2dfglang = {
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
    "PHP": "php",
    "Ruby": "ruby",
}

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
}

# load parsers
parsers: Dict[str, List[object]] = {}
for lang, function in dfg_function.items():
    LANGUAGE = Language(
        "fixme/models/graphcodebert/parser_helpers/my-languages.so", lang
    )
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser_item = [parser, function]
    parsers[lang] = parser_item


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, specific_parser, specific_lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, specific_lang)
    except:
        pass
    # obtain dataflow
    if specific_lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = specific_parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = specific_parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg
