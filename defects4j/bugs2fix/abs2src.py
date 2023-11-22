import sys
import pandas as pd
import os
from tqdm import tqdm
import sys

if len(sys.argv) < 2:
    print("usage:")
    print("python bugs2fix/abs2src.py <path to csv file>")
    exit(1)

csv_path = sys.argv[1]
print(f"Reading from csv {csv_path}.")

df = pd.read_csv(
    csv_path,
    dtype={
        "bid": "string",
        "pid": "string",
        "idx": "string",
        "abstract_path": "string",
        "buggy_path": "string",
        "bugs2fix_path": "string",
        "start_lineno": "int",
        "end_lineno": "int",
    },
)

def read_mapping(path):
    with open(path) as fstream:
        content = fstream.read().split("\n")
    mapping = {} # maps from abstract to original identifier
    for lineno in range(0, len(content), 2):
        if lineno + 1 >= len(content):
            continue
        identifiers = content[lineno].strip().split(",")[:-1]
        abstracts = content[lineno + 1].strip().split(",")[:-1]
        for identifier, abstract in zip(identifiers, abstracts):
            mapping[abstract] = identifier
    #print(mapping)
    return mapping


def abs2src(abs_path, map_path, out_path):
    mapping = read_mapping(map_path)
    try:
        with open(abs_path, "r") as fstream:
            abs = fstream.read()
    except FileNotFoundError:
        print(f"Did not find {abs_path}.")
        return
    src = abs
    for abstract in mapping:
        #print(f"Replacing {abstract} by {mapping[abstract]}")
        src = src.replace(abstract, mapping[abstract])
    with open(out_path, "w") as fstream:
        fstream.write(src)


for _, row in tqdm(df.iterrows(), total=len(df)):
    abs_path = os.path.join(row["abstract_path"], "prediction2")
    map_path = os.path.join(row["abstract_path"], "abstract.map")
    out_path = os.path.join(row["abstract_path"], "concrete2")
    abs2src(abs_path, map_path, out_path)
