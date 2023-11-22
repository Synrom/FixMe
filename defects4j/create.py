import os
from typing import Dict, List, Iterable, Iterator, Tuple
from dataclasses import dataclass
import javalang
from difflib import Differ
import csv
import cchardet
from tqdm import tqdm
import sys

if len(sys.argv) < 4:
    print("usage:")
    print("python create.py <repo_dir> <bugids_dir> <csv_path>")
    exit(1)
    

repo_dir = sys.argv[1] # will be "buggy_repos" after executing initialize.sh
bugids_dir = sys.argv[2] # will be "bugids" after executing initialize.sh
csv_path = sys.argv[3]

print(f"Repository directory {repo_dir}.")
print(f"BugIDs directory {bugids_dir}.")
print(f"Writing results to {csv_path}.")

@dataclass
class Sample:
    encoder_context_until: str
    buggy_segment: str
    encoder_context_from: str
    decoder_context: str
    fixed_segment: str
    start_lineno: int
    end_lineno: int



@dataclass
class Example:
    encoder_context_before: str
    buggy_segment: str
    encoder_context_from: str
    decoder_context: str
    fixed_segment: str
    start_lineno: int
    end_lineno: int
    bid: str
    pid: str
    buggy_path: str
    fixed_path: str
    lang: str = "Java"

@dataclass
class FilePair:
    buggy_content: str
    fixed_content: str
    bid: str
    pid: str
    buggy_path: str
    fixed_path: str

@dataclass
class MethodPair:
    buggy_method: str
    fixed_method: str
    bid: str
    pid: str
    buggy_path: str
    fixed_path: str
    startline: int

@dataclass
class Class:
    bid: str
    pid: str
    class_descriptor: str
    srcdir_buggy: str
    srcdir_fixed: str

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

def decode_safe(data: bytes):
    if data is None:
        return ""
    if encoding := cchardet.detect(data)["encoding"]:
        try:
            return data.decode(encoding=encoding)
        except UnicodeDecodeError as exception:
            print(f"Failed while encoding")
            return ""
    return ""


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
    a: List[str],
    b: List[str],
    diff: List[str],
) -> Iterator[Sample]:
    """Yields func on every changed line."""
    a_lineno, b_lineno = 0, 0
    idx = 0
    sample = None
    while idx < len(diff):
        line = diff[idx]
        if line.startswith("  "):
            # unchanged line
            if sample:
                sample.encoder_context_from = encoder_context_from(a, a_lineno)
                sample.buggy_segment += "\n"
                sample.end_lineno = a_lineno
                yield sample
                sample = None
            a_lineno += 1
            b_lineno += 1
        elif line.startswith("- "):
            # check whether the line was deleted or changed
            if idx < len(diff) - 1 and diff[idx + 1].startswith("+ "):
                # line was changed
                if not sample:
                    decoder_context_sample = decoder_context(b, b_lineno)
                    sample = Sample(
                        encoder_context_until(a, a_lineno),
                        "",
                        "",
                        decoder_context_sample,
                        "",
                        a_lineno,
                        0,
                    )
                sample.fixed_segment = add_to_fix(b, b_lineno, sample.fixed_segment)
                sample.buggy_segment  = add_to_fix(a, a_lineno, sample.buggy_segment)
                idx += 1
                a_lineno += 1
                b_lineno += 1
            else:
                # line was deleted
                if not sample:
                    decoder_context_sample = decoder_context(b, b_lineno)
                    sample = Sample(
                        encoder_context_until(a, a_lineno),
                        "",
                        "",
                        decoder_context_sample,
                        "",
                        a_lineno,
                        0,
                    )
                sample.buggy_segment = add_to_fix(a, a_lineno, sample.buggy_segment)
                a_lineno += 1
        elif line.startswith("+ "):
            # line was added
            if not sample:
                decoder_context_sample = decoder_context(b, b_lineno)
                sample = Sample(
                    encoder_context_until(a, a_lineno),
                    "",
                    "",
                    decoder_context_sample,
                    "",
                    a_lineno,
                    0,
                )
            b_lineno += 1
            sample.fixed_segment = add_to_fix(b, b_lineno, sample.fixed_segment)
        idx += 1


def create_diff(a: List[str], b: List[str]):
    return [
        line
        for line in Differ().compare(a, b)
        if not line.startswith("?") and not len(line) == 0
    ]



def extract_java_methods(code: str) -> Dict[str, Tuple[str, int, int]]:

    def get_method_text(node, codelines: List[str]):
        if not node or not node.children or len(node.children) < 2:
            return "",0,0
        if "abstract" in node.children[1]:
            return "",0,0
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





def methodpair2example(methodpairs: Iterator[MethodPair]) -> Iterator[Example]:
    for methodpair in methodpairs:
        buggy_method = methodpair.buggy_method.splitlines()
        fixed_method = methodpair.fixed_method.splitlines()
        for sample in diff_slicer(buggy_method, fixed_method, create_diff(buggy_method, fixed_method)):
            print(sample.fixed_segment)
            yield Example(
                sample.encoder_context_until,
                sample.buggy_segment,
                sample.encoder_context_from,
                sample.decoder_context,
                sample.fixed_segment,
                sample.start_lineno + methodpair.startline,
                sample.end_lineno + methodpair.startline,
                methodpair.bid,
                methodpair.pid,
                methodpair.buggy_path,
                methodpair.fixed_path,
                "Java",
            )


def filepair2methodpair(filepairs: Iterator[FilePair]) -> Iterator[MethodPair]:
    for filepair in filepairs:
        buggy_methods = extract_java_methods(filepair.buggy_content)
        fixed_methods = extract_java_methods(filepair.fixed_content)

        for name in buggy_methods:
            if not name in fixed_methods or fixed_methods[name][0] == "":
                continue
            if buggy_methods[name][0] == "":
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
                )
    

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
        try:
            with open(buggy_path, "rb") as f:
                buggy_content = decode_safe(f.read())
        except OSError:
            print(f"Could not find file {buggy_path}")
            continue
        try:
            with open(fixed_path, "rb") as f:
                fixed_content = decode_safe(f.read())
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
        )


def class2example(classes: Iterator[Class]) -> List[Example]:
    return [example for example in methodpair2example(filepair2methodpair(class2filepair(classes)))]

def write_examples_to_csv(examples: List[Example]):
    
    rows = [
        (
            example.encoder_context_before,
            example.buggy_segment,
            example.encoder_context_from,
            example.decoder_context,
            example.fixed_segment,
            example.start_lineno,
            example.end_lineno,
            example.pid,
            example.bid,
            example.buggy_path,
            example.fixed_path,
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
            "encoder_context_before",
            "buggy_segment",
            "encoder_context_from",
            "decoder_context",
            "fixed_segment",
            "start_lineno",
            "end_lineno",
            "pid",
            "bid",
            "buggy_path",
            "fixed_path",
        ])


examples: List[Example] = []
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
# examples = []
# buggy = """public class ExampleClass {
# 
#     // Method to calculate the factorial of a number using recursion
#     public static int factorial(int n) {
#         if (n <= 1) {
#             return 1;
#         } else {
#             return n * factorial(n - 1);
#         }
#     }
# 
#     // Method to check if a number is prime
#     public static boolean isPrime(int num) {
#         if (num <= 1) {
#             return false;
#         }
#         for (int i = 2; i <= Math.sqrt(num); i++) {
#             if (num % i == 0) {
#                 return false;
#             }
#         }
#         return true;
#     }
# 
#     public static void main(String[] args) {
#         int number = 5;
#         System.out.println(number + "! = " + factorial(number));
# 
#         int primeNumber = 17;
#         if (isPrime(primeNumber)) {
#             System.out.println(primeNumber + " is a prime number.");
#         } else {
#             System.out.println(primeNumber + " is not a prime number.");
#         }
#     }
# }""".splitlines()
# 
# fixed = """public class ExampleClass {
# 
#     // Method to calculate the factorial of a number using recursion
#     public static int factorial(int n) {
#         return -1;
#     }
# 
#     // Method to check if a number is prime
#     public static boolean isPrime(int num) {
#         for (int i = 2; i <= Math.sqrt(num); i++) {
#             if (num % i == 0) {
#                 return false;
#             }
#         }
#         if (num >= 100) {
#             return false;
#         }
#         return true;
#     }
# 
#     public static void main(String[] args) {
#         int number = 5;
#         System.out.println(number + "! = " + factorial(number));
# 
#         int primeNumber = 17;
#         if (isPrime(primeNumber)) {
#             System.out.println(primeNumber + " is a prime number.");
#         } else {
#             System.out.println(primeNumber + " is not a prime number.");
#         }
#     }
# }""".splitlines()
# 
# 
# for sample in diff_slicer(buggy, fixed, create_diff(buggy, fixed)):
#     print(f"Sample [{sample.start_lineno} : {sample.end_lineno}]")
#     print(buggy[sample.start_lineno:sample.end_lineno])
#     print(sample.buggy_segment)
#     print(sample.fixed_segment)


buggy = """    public LegendItemCollection getLegendItems() {
        LegendItemCollection result = new LegendItemCollection();
        if (this.plot == null) {
            return result;
        }
        int index = this.plot.getIndexOf(this);
        CategoryDataset dataset = this.plot.getDataset(index);
        if (dataset == null) {
            return result;
        }
        int seriesCount = dataset.getRowCount();
        if (plot.getRowRenderingOrder().equals(SortOrder.ASCENDING)) {
            for (int i = 0; i < seriesCount; i++) {
                if (isSeriesVisibleInLegend(i)) {
                    LegendItem item = getLegendItem(index, i);
                    if (item != null) {
                        result.add(item);
                    }
                }
            }
        }
        else {
            for (int i = seriesCount - 1; i >= 0; i--) {
                if (isSeriesVisibleInLegend(i)) {
                    LegendItem item = getLegendItem(index, i);
                    if (item != null) {
                        result.add(item);
                    }
                }
            }
        }
        return result;
    }""".splitlines()

fixed = """    public LegendItemCollection getLegendItems() {
        LegendItemCollection result = new LegendItemCollection();
        if (this.plot == null) {
            return result;
        }
        int index = this.plot.getIndexOf(this);
        CategoryDataset dataset = this.plot.getDataset(index);
        if (dataset != null) {
            return result;
        }
        int seriesCount = dataset.getRowCount();
        if (plot.getRowRenderingOrder().equals(SortOrder.ASCENDING)) {
            for (int i = 0; i < seriesCount; i++) {
                if (isSeriesVisibleInLegend(i)) {
                    LegendItem item = getLegendItem(index, i);
                    if (item != null) {
                        result.add(item);
                    }
                }
            }
        }
        else {
            for (int i = seriesCount - 1; i >= 0; i--) {
                if (isSeriesVisibleInLegend(i)) {
                    LegendItem item = getLegendItem(index, i);
                    if (item != null) {
                        result.add(item);
                    }
                }
            }
        }
        return result;
    }""".splitlines()




#for sample in diff_slicer(buggy, fixed, create_diff(buggy, fixed)):
#    print(f"Sample [{sample.start_lineno} : {sample.end_lineno}]")
#    print(buggy[sample.start_lineno:sample.end_lineno])
#    print(sample.buggy_segment)
#    print(sample.fixed_segment)


