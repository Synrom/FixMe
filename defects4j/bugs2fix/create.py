import os
import cchardet
from typing import Dict, List, Iterable, Iterator, Tuple
from dataclasses import dataclass
import javalang
from difflib import Differ
import csv
from tqdm import tqdm
import sys

if len(sys.argv) < 7:
    print("usage:")
    print("python bugs2fix/create.py <repo_dir> <bugids_dir> <bugs2fix_dir> <idiomfile> <src2abs .jar file> <csv_path>")
    exit(1)

repo_dir = sys.argv[1] # "buggy_repos"
bugids_dir = sys.argv[2] # "bugids"
bugs2fix = sys.argv[3] # "bugs2fix"
idiomfile = sys.argv[4] # system location of idioms.csv from src2abs
abstracter = sys.argv[5] # system location to src2abs .jar file
csv_path = sys.argv[6] # csv that will be created

print(f"Repository directory is {repo_dir}.")
print(f"BugIDs directory is {bugids_dir}.")
print(f"Bugs2Fix directory is {bugs2fix}.")
print(f"Idiom file path is {idiomfile}.")
print(f"Src2abs .jar file path is {abstracter}.")
print(f"Writing results to {csv_path}.")

@dataclass
class Sample:
    buggy_method: str
    start_lineno: int
    end_lineno: int


@dataclass
class AbstractExample:
    start_lineno: int
    end_lineno: int
    bid: str
    pid: str
    buggy_path: str
    idx: int
    abstract_path: str
    bugs2fix_path: str


@dataclass
class Example:
    buggy_method: str
    start_lineno: int
    end_lineno: int
    bid: str
    pid: str
    buggy_path: str
    fixed_path: str
    bugs2fix_path: str

@dataclass
class FilePair:
    buggy_content: str
    fixed_content: str
    bid: str
    pid: str
    buggy_path: str
    fixed_path: str
    bugs2fix_path: str

@dataclass
class MethodPair:
    buggy_method: str
    fixed_method: str
    bid: str
    pid: str
    buggy_path: str
    fixed_path: str
    start_lineno: int
    end_lineno: int
    bugs2fix_path: str

@dataclass
class Class:
    bid: str
    pid: str
    class_descriptor: str
    srcdir_buggy: str
    srcdir_fixed: str


def extract_java_methods(code: str) -> Dict[str, Tuple[str, int, int]]:

    def get_method_text(node, codelines: List[str]):
        if not node or not node.children or len(node.children) < 2:
            return "", 0, 0
        if "abstract" in node.children[1]:
            return "", 0, 0
        startline = node.position.line - 1
        endline = startline
        if ender := node.children[-1]:
            while True:
                if isinstance(ender, list) and len(ender) > 0:
                    ender = ender[-1]
                elif isinstance(ender, javalang.tree.Node):
                    if ender.position:
                        endline = ender.position.line + 1
                    if ender.children:
                        ender = ender.children
                    else:
                        break
                else:
                    break
        
        if endline != startline:
            while endline < len(codelines) and (codelines[endline].strip() == "" or (codelines[endline].strip().startswith("}") and not codelines[endline].startswith("}"))):
                endline += 1
        else:
            endline = startline + 2
        return "\n".join(codelines[startline:endline]), startline, endline

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


def abstract2dir(abs: AbstractExample) -> str:
    path = os.path.join(
        bugs2fix,
        abs.pid,
        abs.bid,
        str(abs.idx),
    )
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def example2abstract(examples: Iterator[Example]) -> Iterator[AbstractExample]:
    for idx, example in enumerate(examples):
        abs = AbstractExample(
            example.start_lineno,
            example.end_lineno,
            example.bid,
            example.pid,
            example.buggy_path,
            idx,
            "",
            example.bugs2fix_path,
        )
        base = abstract2dir(abs)
        abs.abstract_path = base

        source = os.path.join(base, "source.java")
        with open(source, "w") as f:
            f.write(example.buggy_method)

        path = os.path.join(base, "abstract")
        cmd = "java -jar " + abstracter + " single method "+source+" "+path+" "+idiomfile
        print(cmd)
        os.system(cmd)
        yield abs




def methodpair2example(methodpairs: Iterator[MethodPair]) -> Iterator[Example]:
    for methodpair in methodpairs:
        buggy_method = methodpair.buggy_method
        fixed_method = methodpair.fixed_method
        if buggy_method != fixed_method:
            yield Example(
                buggy_method,
                methodpair.start_lineno,
                methodpair.end_lineno,
                methodpair.bid,
                methodpair.pid,
                methodpair.buggy_path,
                methodpair.fixed_path,
                methodpair.bugs2fix_path,
            )

def filepair2methodpair(filepairs: Iterator[FilePair]) -> Iterator[MethodPair]:
    for filepair in filepairs:
        print(filepair.buggy_path)
        buggy_methods = extract_java_methods(filepair.buggy_content)
        fixed_methods = extract_java_methods(filepair.fixed_content)

        for name in buggy_methods:
            if name not in fixed_methods:
                continue
            if fixed_methods[name][0] == "" or buggy_methods[name][0] == "":
                continue
            if fixed_methods[name][0] != buggy_methods[name][0]:
                yield MethodPair(
                    buggy_methods[name][0],
                    fixed_methods[name][0],
                    filepair.bid,
                    filepair.pid,
                    filepair.buggy_path,
                    filepair.fixed_path,
                    buggy_methods[name][1],
                    buggy_methods[name][2],
                    filepair.bugs2fix_path,
                )

    

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


def class2filepair(classes: Iterator[Class]) -> Iterator[FilePair]:
    for class_object in classes:
        class_path = class_object.class_descriptor.replace(".", "/")
        buggy_path = os.path.join(
            repo_dir,
            class_object.pid,
            class_object.bid,
            "buggy",
            class_object.srcdir_buggy,
            f"{class_path}.java"
        )
        fixed_path = os.path.join(
            repo_dir,
            class_object.pid,
            class_object.bid,
            "fixed",
            class_object.srcdir_fixed,
            f"{class_path}.java"
        )
        bugs2fix_path = os.path.join(
            repo_dir,
            class_object.pid,
            class_object.bid,
            "bugs2fix",
            class_object.srcdir_fixed,
            f"{class_path}.java"
        )
        try:
            with open(buggy_path, "rb") as f:
                buggy_content = decode_safe(f.read())
                if buggy_content == "":
                    continue
        except OSError:
            print(f"Could not find file {buggy_path}")
            continue
        try:
            with open(fixed_path, "rb") as f:
                fixed_content = decode_safe(f.read())
                if fixed_content == "":
                    continue
        except OSError:
            print(f"Could not find file {buggy_path}")
            continue
        
        yield FilePair(
            buggy_content, 
            fixed_content,
            class_object.bid,
            class_object.pid,
            buggy_path,
            fixed_path,
            bugs2fix_path,
        )


def class2example(classes: Iterator[Class]) -> List[AbstractExample]:
    return [example for example in example2abstract(methodpair2example(filepair2methodpair(class2filepair(classes))))]

def write_examples_to_csv(examples: List[AbstractExample]):
    
    rows = [
        (
            example.bid,
            example.pid,
            example.idx,
            example.abstract_path,
            example.buggy_path,
            example.bugs2fix_path,
            example.start_lineno,
            example.end_lineno,
        ) 
        for example in examples
    ]

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def init_csv():
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "bid",
            "pid",
            "idx",
            "abstract_path",
            "buggy_path",
            "bugs2fix_path",
            "start_lineno",
            "end_lineno",
        ])


examples: List[AbstractExample] = []
total = len([filename for filename in os.listdir(bugids_dir) if filename.endswith(".buggy")])
init_csv()

with tqdm(total=total) as pbar:
    for filename in os.listdir(bugids_dir):
        if not filename.endswith(".classes"):
            continue
        path = os.path.join(bugids_dir, filename)
        repository = filename[:filename.find(".classes")]
        with open(path, "r") as f:
            for line in f.readlines():
                bugid, classes_str = line.split(",")
                classes_str = classes_str.strip()
                # delete quotes
                classes_str = classes_str[classes_str.find('"')+1:]
                classes_str = classes_str[:classes_str.find('"')]
                classes = classes_str.split(";")
                pbar.update()
                try:
                    with open(os.path.join(bugids_dir, f"{repository}_{bugid}.buggy")) as f:
                        srcdir_buggy = f.read().strip()
                except OSError:
                    print(f"Could not find file {repository}_{bugid}.buggy")
                    continue
                try:
                    with open(os.path.join(bugids_dir, f"{repository}_{bugid}.fixed")) as f:
                        srcdir_fixed = f.read().strip()
                except OSError:
                    print(f"Could not find file {repository}_{bugid}.fixed")
                    continue
                class_objects = [Class(bugid, repository, class_descriptor, srcdir_buggy, srcdir_fixed) for class_descriptor in classes]
                examples += class2example(iter(class_objects))
        if len(examples) >= 1000:
            write_examples_to_csv(examples)
            examples = []

write_examples_to_csv(examples)
examples = []
