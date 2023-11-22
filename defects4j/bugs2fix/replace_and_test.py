import pandas as pd
import os
from tqdm import tqdm
import javalang
import cchardet
import sys

if len(sys.argv) < 2:
    print("usage:")
    print("python bugs2fix/replace_and_test.py <path to csv file>")
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

paths_to_test = set()

syntax_right = 0
syntax_false = 0

def decode_safe(data: bytes):
    if data is None:
        return ""
    if encoding := cchardet.detect(data)["encoding"]:
        try:
            return data.decode(encoding=encoding)
        except UnicodeDecodeError as exception:
            print(exception)
            return ""
    return ""

with open("progress", "w") as f:
    pass

changes = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row["bugs2fix_path"]
    startline = row["start_lineno"]
    endline = row["end_lineno"]
    try:
        with open(os.path.join(row["abstract_path"], "concrete2"), "rb") as fstream:
            prediction = decode_safe(fstream.read())
            if prediction == "":
                continue
    except FileNotFoundError:
        print(f"Could not find {os.path.join(row['abstract_path'], 'concrete2')}")
        continue

    bugs2fix = os.path.join(
        "buggy_repos",
        row["pid"],
        row["bid"],
        "bugs2fix"
    )

    if bugs2fix in changes:
        changes[bugs2fix].append((prediction, startline, endline, path))
    else:
        changes[bugs2fix] = [(prediction, startline, endline, path)]


paths_to_test = set()
for bugs2fix in changes:
    
    # sort changes by startline
    schanges = sorted(changes[bugs2fix], key=lambda x:x[1], reverse=True)
    paths_to_test.add(bugs2fix)

    for prediction, startline, endline, path in schanges:
        with open(path, "rb") as fstream:
            codelines = decode_safe(fstream.read()).splitlines()
            if codelines == []:
                continue
        
        before = "\n".join(codelines[:startline])
        after = "\n".join(codelines[endline:])
        code= before + "\n" + prediction + "\n" + after

        with open(path, "w") as fstream:
            fstream.write(code)
        
        with open("progress", "a") as f:
            f.write(row["abstract_path"] + "\n")

        try:
            tree = javalang.parse.parse(code)
        except javalang.parser.JavaSyntaxError:
            syntax_false += 1
            paths_to_test.remove(bugs2fix)
            break
        except Exception as exception:
            print(exception)
        
        syntax_right += 1

with open("log", "w") as f:
    f.write(f"{syntax_right} samples were syntactically correct\n")
    f.write(f"{syntax_false} samples were syntactically wrong")

def handle(path):
    os.chdir(path)
    os.system("defects4j compile")
    os.system("timeout 10m defects4j test")
    os.chdir("/home/synrom/2023_ba_swierzy_leiwig/dataset/scripts/defects4j")

print("Starting to compile")
for path in tqdm(paths_to_test, total=len(paths_to_test)):
    handle(path)
