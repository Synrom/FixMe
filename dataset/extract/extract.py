"""Extract code samples."""
import sys
import os
from github import Github
from typing import Any, Iterable, List, Tuple, Dict, Optional
from tqdm import tqdm
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
from phply.phpparse import make_parser
from phply.phplex import lexer as php_lexer
import phply.phpast as php_ast

php_parser = make_parser()

def parse_php(code):
    lexer = php_lexer
    lexer.lineno = 1

    try:
        result = php_parser.parse(code, lexer=lexer.clone(), tracking=True)
    except:
        return []  

    php_parser.restart()
    return result


def extract_php_methods(code: str) -> Dict[str, str]:
    result = parse_php(code)
    code = code.splitlines()
    methods = {}
    method_name = None
    start_line = None
    end_line = None

    def crawl_ast(node):
        nonlocal method_name
        nonlocal start_line
        nonlocal end_line
        nonlocal code
        if method_name:
            try:
                end_line = node.lineno - 1
                methods[method_name] = "\n".join(code[start_line-1:end_line])
            except AttributeError:
                pass
            method_name = None
        if isinstance(node, php_ast.Function) or isinstance(node, php_ast.Method):
            start_line = node.lineno
            for child in node.nodes:
                crawl_ast(child)
            method_name = node.name
            return
        if getattr(node, "nodes", None):
            for child in node.nodes:
                crawl_ast(child)
    for node in result:
        crawl_ast(node)
    return methods

def extract_python_methods(code: str) -> Dict[str, str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return {}

    methods = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = ast.get_source_segment(code, node)
    return methods


def extract_javascript_methods(code: str) -> Dict[str, str]:
    try:
        tree = esprima.parseScript(code, options={"range": True})
    except Exception:
        return {}

    context = {}

    class ContextExtractor(esprima.NodeVisitor):
        def visit_FunctionDeclaration(self, node):
            name = node.id.name
            start, finish = node.range
            nonlocal code, context
            context[name] = code[start:finish]
        def visit_MethodDefinition(self, node):
            name = node.key.name
            start, finish = node.range
            nonlocal code, context
            context[name] = code[start:finish]

    ContextExtractor().visit(tree)
    return context

def extract_ruby_methods(code: str) -> Dict[str, str]:
    functions = {}
    function_name: Optional[str] = None
    function_code: List[str] = []
    for line in code.splitlines():
        line = line.strip()

        if line.startswith('def '):
            function_name = line.split()[1].split("(")[0]
            function_code = []

        if line.startswith('end') and function_name:
            function_code.append(line)
            functions[function_name] = "\n".join(function_code)
            function_name = None

        if function_name:
            function_code.append(line)

    return functions


def extract_java_methods(code: str) -> Dict[str, str]:
    if code.count("\n") > 2000:
        return {}

    def get_method_text(node, codelines: List[str]):
        if not node or not node.children or len(node.children) < 2:
            return ""
        if "abstract" in node.children[1]:
            return ""
        startline = node.position.line - 1
        if node.children[-1] and len(node.children[-1]) > 0:
            endline = node.children[-1][-1].position.line + 1
        else:
            endline = startline + 2
        return "\n".join(codelines[startline:endline])

    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return {}
    context = {}
    try:
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            name = node.name
            context[name] = get_method_text(node, code.splitlines())
    except RecursionError:
        print("Got Recursion error while parsing.")
    return context


def create_samples(
    before: str,
    after: str,
    language: str,
) -> List[Tuple[str, str]]:
    if language == "Python":
        extractor = extract_python_methods
    elif language == "JavaScript":
        extractor = extract_javascript_methods
    elif language == "Java":
        extractor = extract_java_methods
    elif language == "Ruby":
        extractor = extract_ruby_methods
    elif language == "PHP":
        extractor = extract_php_methods
    else:
        return []
    buggy_methods = extractor(before)
    fixed_methods = extractor(after)
    if buggy_methods == {} or fixed_methods == {}:
        return []
    samples = []
    for method_name, buggy_method in buggy_methods.items():
        if not isinstance(method_name, str) or "test" in method_name:
            continue
        try:
            if fixed_methods[method_name] != buggy_method:
                samples.append((buggy_method, fixed_methods[method_name]))
        except KeyError:
            continue
    return samples


def write_samples_to_csv(
    filename: str, data: List[Tuple[str, str, str, str, str, str, str, str, str, str]]
):
    with open(filename, "a", newline="") as csvfile:
        csv.writer(csvfile).writerows(data)


def decode_safe(data: bytes):
    if data is None:
        return ""
    if encoding := cchardet.detect(data)["encoding"]:
        try:
            return data.decode(encoding=encoding)
        except UnicodeDecodeError as exception:
            logging.exception(exception)
            return ""
    return ""


def determine_language(path):
    if path.endswith(".py"):
        return "Python"
    if path.endswith(".js"):
        return "JavaScript"
    if path.endswith(".java"):
        return "Java"
    if path.endswith(".rb"):
        return "Ruby"
    if path.endswith(".php"):
        return "PHP"
    return "Unknown"


def get_commit_date(repo: Repo, commit_hash: str) -> str:
    commit = repo.commit(commit_hash)
    return str(commit.committed_datetime)


def process_row(repo, row) -> List[Tuple[str, str, str, str, str, str, str, str, str, str]]:
    author, repo_name = row["repo_name"].split("/")
    commit_hash = row["commit"]
    watch_count = row["watch_count"]

    try:
        commit = repo.commit(commit_hash)
        commit_date = get_commit_date(repo, commit_hash)
    except Exception as exception:
        logging.error(exception)
        return []

    samples = []

    for diff in commit.diff(commit.parents):
        if any([not diff.a_blob, not diff.b_blob, not diff.a_path, not diff.b_path]):
            continue

        language = determine_language(diff.a_path)
        if language == "Unknown" or language != determine_language(diff.b_path):
            continue

        try:
            after = decode_safe(diff.a_blob.data_stream.read())
            before = decode_safe(diff.b_blob.data_stream.read())
            if before == "" or after == "":
                continue
        except Exception as exception:
            print(f"while decoding had exception {exception}")
            continue

        for buggy, fixed in create_samples(before, after, language):
            sample = (
                buggy,
                fixed,
                author,
                repo_name,
                commit_hash,
                language,
                commit_date,
                watch_count,
                diff.a_path,
                diff.b_path,
            )
            samples.append(sample)
    return samples


repositories_path = sys.argv[1]
commits_path = sys.argv[2]

def process_repo(data: Tuple[str, int]):
    """Processes one repositorie."""
    name, nmr = data
    # print(f"Start processing repository {name}.")
    path = f"repositories/{name}"
    try:
        repo = Repo(path)
    except Exception as exception:
        print(f"error while creating repo {name}.")
        return
    df = pd.read_csv(
        commits_path,
        dtype={
            "commit": "string",
            "subject": "string",
            "message": "string",
            "repo_name": "string",
            "language": "string",
            "watch_count": "int",
        },
    )
    commits = df[df["repo_name"] == f"{name}"]
    samples = []
    idx = 0
    csv_path = f"method_pairs/{name.replace('/', '_')}.csv"
    for _, row in commits.iterrows():
        samples += process_row(repo, row)
        if len(samples) >= 1000:
            write_samples_to_csv(csv_path, samples)
            samples = []
        idx += 1
    write_samples_to_csv(csv_path, samples)
    #print(f"Done processing {nmr}th repo {name}.")

if not os.path.isdir("method_pairs"):
    os.mkdir("method_pairs")

print("Extract method pairs to method_pairs/.")
print(f"Repositories are taken from {repositories_path}.")
print(f"Commits are taken from {commits_path}.")

repos = pd.read_csv(
    repositories_path,
    dtype={"name": "string", "language": "string", "commits": "int", "watch_count": "int"},
)
repo_names = repos["name"].unique()
count = [i for i in range(1, len(repo_names) + 1)]
print(f"have {cpu_count()} cpus")
with Pool(processes=cpu_count()) as pool:
    with tqdm(total=len(repo_names)) as pbar:
        for _ in pool.imap_unordered(process_repo, zip(repo_names, count)):
            pbar.update()
#   pool.map(
#       process_repo,
#       zip(repo_names, count),
#   )

js_code = """
function calculateFactorial(n) {
    if (n === 0 || n === 1) {
        return 1;
    } else {
        let factorial = 1;
        for (let i = 2; i <= n; i++) {
            factorial *= i;
        }
        return factorial;
    }
}

function generateFibonacciSequence(n) {
    let sequence = [0, 1];
    if (n <= 2) {
        return sequence.slice(0, n);
    } else {
        for (let i = 2; i < n; i++) {
            let nextNumber = sequence[i - 1] + sequence[i - 2];
            sequence.push(nextNumber);
        }
        return sequence;
    }
}

const fibonacciCalculator = {
  calculateFibonacci: function(n) {
    let fibonacci = [0, 1];

    if (n <= 1) {
      return fibonacci.slice(0, n + 1);
    }

    for (let i = 2; i <= n; i++) {
      fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }

    return fibonacci;
  }
};
"""
# for sample in extract_javascript_context(js_code).items():
#     print(sample[0])
#     print(sample[1])

python_code = """
def calculate_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        factorial = 1
        for i in range(2, n + 1):
            factorial *= i
        return factorial

def generate_fibonacci_sequence(n):
    sequence = [0, 1]
    if n <= 2:
        return sequence[:n]
    else:
        for i in range(2, n):
            next_number = sequence[i - 1] + sequence[i - 2]
            sequence.append(next_number)
        return sequence
"""
# for sample in extract_python_methods(python_code).items():
#     print(sample[0])
#     print(sample[1])

java_code = """
public class MathUtils {

    public static int calculateFactorial(int n) {
        if (n == 0 || n == 1) {
            return 1;
        } else {
            int factorial = 1;
            for (int i = 2; i <= n; i++) {
                factorial *= i;
            }
            return factorial;
        }
    }

    public static int[] generateFibonacciSequence(int n) {
        int[] sequence = new int[n];
        sequence[0] = 0;
        sequence[1] = 1;
        if (n <= 2) {
            return sequence;
        } else {
            for (int i = 2; i < n; i++) {
                sequence[i] = sequence[i - 1] + sequence[i - 2];
            }
            return sequence;
        }
    }

    public int[] generateFibonacciSequenceMethod(int n) {
        int[] sequence = new int[n];
        sequence[0] = 0;
        sequence[1] = 1;
        if (n <= 2) {
            return sequence;
        } else {
            for (int i = 2; i < n; i++) {
                sequence[i] = sequence[i - 1] + sequence[i - 2];
            }
            return sequence;
        }
    }
}

"""
# for sample in extract_java_context(java_code).items():
#     print(sample[0])
#     print(sample[1])
#
# print(extract_python_context(python_code.split("\n")))
# print(extract_javascript_context(js_code.split("\n")))
# print(extract_java_context(java_code.split("\n")))
# print("First context:")
# print(extract_python_context(python_code.split("\n"), 4))
# print("Second context:")
# print(extract_python_context(python_code.split("\n"), 13))
# print("First context:")
# print(extract_javascript_context(js_code.split("\n"), 4))
# print("Second context:")
# print(extract_javascript_context(js_code.split("\n"), 13))
# print("Third context:")
# print(extract_javascript_context(js_code.split("\n"), 30))
# print("First context:")
# print(extract_java_context(java_code.split("\n"), 7))
# print("Second context:")
# print(extract_java_context(java_code.split("\n"), 17))
# print("Third context:")
# print(extract_java_context(java_code.split("\n"), 31))

# ruby_code = """
# def example_method
#   # Method body
# end
# 
# def another_method
#   # Method body
# end
# def greet(name)
#   if name.nil? || name.empty?
#     return "Hello, anonymous!"
#   else
#     return "Hello, #{name}!"
#   end
# end
# 
# puts greet("Alice")
# puts greet("")
# class Sample
#    def hello
#       puts "Hello Ruby!"
#    end
# end
# 
# # Now using above class to create objects
# object = Sample. new
# object.hello
# """
# methods = extract_ruby_methods(ruby_code)
# for method in methods:
#     print(method)
#     print(methods[method])


# test_repo_name = "apache/isis"
# test_commit_hash = "6427dfb1a4f2296c6aeaf6bf240af8a0cd2822cd"
# test_watch_count = 45
# test_df = pd.DataFrame(
#     {
#         "repo_name": [test_repo_name],
#         "commit": [test_commit_hash],
#         "watch_count": [test_watch_count],
#     },
# )
# # g = Github("secret")
# # repository = g.get_repo(test_repo_name)
# # 
# # # Clone the repository using GitPython or perform any other desired actions
# # clone_url = repository.clone_url
# # 
# # path = f"repositories/{test_repo_name}"
# # Repo.clone_from(
# #     clone_url, path, allow_unsafe_options=True, allow_unsafe_protocols=True
# # )
# repo = Repo("repositories/apache/isis")
# 
# process_row(repo, test_df.iloc[0])

# test_repo_name = "ChromeDevTools/devtools-frontend"
# test_commit_hash = "00e15811f59eba6d923c8338af3dca6bb37cf68c"
# test_watch_count = 79
# test_df = pd.DataFrame(
#     {
#         "repo_name": [test_repo_name],
#         "commit": [test_commit_hash],
#         "watch_count": [test_watch_count],
#     },
# )
# g = Github("")
# repository = g.get_repo(test_repo_name)
# 
# # Clone the repository using GitPython or perform any other desired actions
# clone_url = repository.clone_url
# 
# path = f"repositories/{test_repo_name}"
# Repo.clone_from(
#     clone_url, path, allow_unsafe_options=True, allow_unsafe_protocols=True
# )
# repo = Repo("repositories/apache/isis")
# 
# process_row(repo, test_df.iloc[0])
