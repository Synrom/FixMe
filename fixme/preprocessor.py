from typing import Callable
from .example import Example

PreprocessorFn = Callable[[Example], Example]


def delete_double_spaces(sample: Example) -> Example:
    """Recursively Replaces all double spaces with single spaces."""

    def clear_string(string: str) -> str:
        while "  " in string:
            string = string.replace("  ", " ")
        return string

    sample.buggy_segment = clear_string(sample.buggy_segment)
    sample.source_from = clear_string(sample.source_from)
    sample.source_until = clear_string(sample.source_until)
    sample.target = clear_string(sample.target)
    sample.target_context = clear_string(sample.target_context)
    return sample

def delete_indentation(sample: Example) -> Example:
    if sample.lang == "Python":
        return sample
    def clear_string(string: str) -> str:
        lines = string.splitlines()
        resulting_lines = []
        for line in lines:
            position = 0
            length = len(line)
            while position < length and line[position] in [" ", "\t"]:
                position += 1
            resulting_lines.append(line[position:])
        return "\n".join(resulting_lines)
    sample.buggy_segment = clear_string(sample.buggy_segment)
    sample.source_from = clear_string(sample.source_from)
    sample.source_until = clear_string(sample.source_until)
    sample.target = clear_string(sample.target)
    sample.target_context = clear_string(sample.target_context)
    return sample
    

def no_context(sample: Example) -> Example:
    sample.source_from = ""
    sample.source_until = ""
    sample.target_context = ""
    sample = delete_indentation(sample)
    return sample



def no_preprocessing(sample: Example) -> Example:
    return sample
