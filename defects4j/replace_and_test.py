
import pandas as pd
import os
from tqdm import tqdm
import javalang
import cchardet
import pathlib

current_dir = str(pathlib.Path().resolve())

if len(sys.argv) < 3:
    print("usage:")
    print("python replace_and_test <repo_dir> <csv_path>")
    exit(1)
    

repo_dir = sys.argv[1] # will be "buggy_repos" after executing initialize.sh
csv_path = sys.argv[2]

print(f"Repository directory {repo_dir}.")
print(f"Reading fixes from {csv_path}.")

df = pd.read_csv(
    csv_path,
    dtype={
        "encoder_context_before": "string",
        "buggy_segment": "string",
        "encoder_context_from": "string",
        "decoder_context": "string",
        "fixed_segment": "string",
        "start_lineno": "int",
        "end_lineno": "int",
        "pid": "string",
        "bid": "string",
        "buggy_path": "string",
        "fixed_path": "string",
        "prediction": "string",
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

changes = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    base_dir = f"{repo_dir}/{row['pid']}/{row['bid']}"


    test_dir = base_dir + "/unixcoder"
    buggy_dir = base_dir + "/buggy"

    buggy_file = row["buggy_path"][len(buggy_dir):]
    buggy_file = test_dir + "/" + buggy_file
    # print(buggy_file)
    # print(buggy_dir)
    # print(test_dir)

    startline = row["start_lineno"]
    endline = row["end_lineno"]

    prediction = row["prediction"].replace("\\n", "\n")

    if test_dir in changes:
        changes[test_dir].append((buggy_file, startline, endline, prediction))
    else:
        changes[test_dir] = [(buggy_file, startline, endline, prediction)]


paths_to_test = set()
for buggy_dir in changes:
    
    # sort changes by startline
    schanges = sorted(changes[buggy_dir], key=lambda x:x[1], reverse=True)
    paths_to_test.add(buggy_dir)

    for buggy_file, startline, endline, prediction in schanges:
        with open(buggy_file, "rb") as fstream:
            codelines = decode_safe(fstream.read()).splitlines()
            if codelines == []:
                continue
        
        before = "\n".join(codelines[:startline])
        after = "\n".join(codelines[endline:])
        code= before + "\n" + prediction + "\n" + after

        # print("before:")
        # print("\n".join(codelines[:startline][-10:]))
        # print("prediction:")
        # print(prediction)
        # print("after:")
        # print("\n".join(codelines[endline:][:10]))


        with open(buggy_file, "w") as fstream:
            fstream.write(code)
        
        try:
            tree = javalang.parse.parse(code)
        except javalang.parser.JavaSyntaxError:
            syntax_false += 1
            paths_to_test.remove(buggy_dir)
            break
        except Exception as exception:
            print(exception)
        
        syntax_right += 1


with open("log_unixcoder", "w") as f:
    f.write(f"{syntax_right} samples were syntactically correct\n")
    f.write(f"{syntax_false} samples were syntactically wrong")

def handle(path):
    os.chdir(path)
    os.system("defects4j compile")
    os.system("timeout 10m defects4j test")
    os.chdir(current_dir)

print("Starting to compile")
for path in tqdm(paths_to_test, total=len(paths_to_test)):
    handle(path)
