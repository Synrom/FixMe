"""Extract single line code samples."""
from typing import Any, Iterable, List, Tuple
from tqdm import tqdm
import os
import logging
import cchardet
from difflib import Differ
import csv
from git import Repo
import pandas as pd
from blake3 import blake3
from multiprocessing import Pool, current_process, Manager, cpu_count
from functools import partial
import ast
import esprima
import javalang
from pyjsparser import parse
from dataclasses import dataclass


@dataclass
class Sample:
    encoder_context_until: str
    buggy_segment: str
    encoder_context_from: str
    decoder_context: str
    fixed_segment: str


def encoder_context_until(
    before: List[str],
    before_line: int,
) -> str:
    encoder_context_until = "\n".join(before[:before_line]) + "\n"
    return encoder_context_until

def encoder_context_from(
    before: List[str],
    before_line: int,
) -> str:
    encoder_context_from = "\n".join(before[before_line:])
    return encoder_context_from


def decoder_context(
    after: List[str],
    after_line: int,
) -> str:
    decoder_context = "\n".join(after[:after_line]) + "\n"
    return decoder_context


def add_to_fix(
    after: List[str],
    after_line: int,
    fix: str = "",
) -> str:
    if after_line >= len(after):
        return fix
    if fix == "":
        fix = after[after_line]
    else:
        fix += "\n" + after[after_line]
    return fix


def diff_slicer(
    buggy: List[str],
    fixed: List[str],
    diff: List[str],
) -> Iterable[Sample]:
    """Yields func on every changed line."""
    buggy_lineno, fixed_lineno = 0, 0
    idx = 0
    sample = None
    while idx < len(diff):
        line = diff[idx]
        if line.startswith("  "):
            # unchanged line
            if sample:
                sample.encoder_context_from = encoder_context_from(buggy, buggy_lineno)
                sample.buggy_segment += "\n"
                yield sample
                sample = None
            buggy_lineno += 1
            fixed_lineno += 1
        elif line.startswith("- "):
            # check whether the line was deleted or changed
            if idx < len(diff) - 1 and diff[idx + 1].startswith("+ "):
                # line was changed
                if not sample:
                    decoder_context_sample = decoder_context(fixed, fixed_lineno)
                    sample = Sample(
                        encoder_context_until(buggy, buggy_lineno),
                        "",
                        "",
                        decoder_context_sample,
                        "",
                    )
                sample.fixed_segment = add_to_fix(fixed, fixed_lineno, sample.fixed_segment)
                sample.buggy_segment  = add_to_fix(buggy, buggy_lineno, sample.buggy_segment)
                idx += 1
                buggy_lineno += 1
                fixed_lineno += 1
            else:
                # line was deleted
                if not sample:
                    decoder_context_sample = decoder_context(fixed, fixed_lineno)
                    sample = Sample(
                        encoder_context_until(buggy, buggy_lineno),
                        "",
                        "",
                        decoder_context_sample,
                        "",
                    )
                sample.buggy_segment = add_to_fix(buggy, buggy_lineno, sample.buggy_segment)
                buggy_lineno += 1
        elif line.startswith("+ "):
            # line was added
            if not sample:
                decoder_context_sample = decoder_context(fixed, fixed_lineno)
                sample = Sample(
                    encoder_context_until(buggy, buggy_lineno),
                    "",
                    "",
                    decoder_context_sample,
                    "",
                )
            sample.fixed_segment = add_to_fix(fixed, fixed_lineno, sample.fixed_segment)
            fixed_lineno += 1
        else:
            logging.error("Line doesn't have a valid start: '%s'", line)
        idx += 1


def create_diff(a: List[str], b: List[str]):
    return [
        line
        for line in Differ().compare(a, b)
        if not line.startswith("?") and not len(line) == 0
    ]


def write_samples_to_csv(
    filename: str, data: List[Tuple[str, str, str, str, str, str, str, str, str, str, str]]
):
    with open(filename, "a", newline="") as csvfile:
        csv.writer(csvfile).writerows(data)


def calc_hash(s1: str, s2: str, s3: str, s4: str, s5: str):
    blake3((s1 + s2 + s3 + s4 + s5).encode("utf-8")).hexdigest()


def process_row(row) -> List[Tuple[str, str, str, str, str, str, str, str, str, str, str]]:
    try:
        if "test" in row["a_path"].lowercase() or "test" in row["b_path"].lowercase():
            return []
    except Exception:
        pass
    try:
        a = row["buggy"].splitlines()
        b = row["fixed"].splitlines()
    except Exception:
        return []

    samples: List[Tuple[str, str, str, str, str, str, str, str, str, str, str]] = [
        (
            sample.encoder_context_until,
            sample.buggy_segment,
            sample.encoder_context_from,
            sample.decoder_context,
            sample.fixed_segment,
            row["author"],
            row["repo_name"],
            row["commit_hash"],
            row["language"],
            row["commit_date"],
            row["watch_count"],
            calc_hash(
                sample.encoder_context_until,
                sample.buggy_segment,
                sample.encoder_context_from,
                sample.decoder_context,
                sample.fixed_segment,
            ),
        )
        for sample in diff_slicer(a, b, create_diff(a, b))
    ]
    return samples


def process_csv(data: Tuple[Tuple[str, str], int]):
    """Processes one repositorie."""
    filenames, nmr = data
    old_path, new_path = filenames
    # print(f"Start processing csv {old_path}.")
    df = pd.read_csv(
        old_path,
        names=[
            "buggy",
            "fixed",
            "author",
            "repo_name",
            "commit_hash",
            "language",
            "commit_date",
            "watch_count",
            "a_path",
            "b_path",
        ],
        dtype={
            "buggy": "string",
            "fixed": "string",
            "author": "string",
            "repo_name": "string",
            "commit_hash": "string",
            "language": "string",
            "commit_date": "string",
            "watch_count": "int",
            "a_path": "string",
            "b_path": "string",
        }
    )
    samples = []
    idx = 0
    for _, row in df.iterrows():
        samples += process_row(row)
        if len(samples) >= 1000:
            write_samples_to_csv(new_path, samples)
            samples = []
        idx += 1
    write_samples_to_csv(new_path, samples)
    #print(f"Done processing {nmr}th repo {old_path}.")

if not os.path.isdir("segment_pairs"):
    os.mkdir("segment_pairs")

print("Creating segment pairs..")
filenames = [
    (f"method_pairs_beautified/{path}", f"segment_pairs/{path}")
    for path in os.listdir("method_pairs_beautified")
]
count = [i for i in range(1, len(filenames) + 1)]
with Pool(processes=cpu_count()) as pool:
   with tqdm(total=len(filenames)) as pbar:
       for _ in pool.imap_unordered(process_csv, zip(filenames, count)):
           pbar.update()
#   pool.map(
#      process_csv,
#      zip(filenames, count),
#  )
# buggy_python = """
# public class Calculator {
#     public static int divideNumbers(int a, int b) {
#         int result = a / b;
#         System.out.println("just a little test");
#         return result;
#     }
# }
# """.splitlines()
# fixed_python = """
# public class Calculator {
#     public static int divideNumbers(int a, int b) {
#             throw new IllegalArgumentException("Cannot divide by zero.");
#         }
#         int result = a / b;
#         return result;
#         int abitofstufadded = a/b;
#         return a * 10;
#     }
# }
# """.splitlines()
# print(create_diff(buggy_python, fixed_python))
# for sample in diff_slicer(
#     buggy_python, fixed_python, create_diff(buggy_python, fixed_python)
# ):
#     print("encoder context until:")
#     print(sample.encoder_context_until)
#     print("buggy_segment:")
#     print(sample.buggy_segment)
#     print("encoder context from:")
#     print(sample.encoder_context_from)
#     print("decoder context:")
#     print(sample.decoder_context)
#     print("fix:")
#     print(sample.fixed_segment)
#     print("encoder whole:")
#     print(sample.encoder_context_until + sample.buggy_segment + sample.encoder_context_from)
#     print("decoder whole:")
#     print(sample.decoder_context + sample.fixed_segment)
