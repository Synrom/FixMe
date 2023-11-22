import re
import os
import pandas as pd
from multiprocessing import Pool, current_process, Manager, cpu_count
from blake3 import blake3
from tqdm import tqdm

def is_different(buggy: str, fixed: str):
    if buggy[:len(fixed)] == fixed:
        return False
    return True

def calculate_hash(row):
    return blake3(
        (
            row["encoder_context_until"]
            + row["buggy_segment"]
            + row["encoder_context_from"]
            + row["decoder_context"]
            + row["fixed_segment"]
        ).encode("utf-8")
    ).hexdigest()

def check_it(row):
    hash = calculate_hash(row)
    if type(hash) != str:
        print("this is weird")
    return hash

def remove_comments_javascript(js_string):
    pattern = r'//.*?$|/\*.*?\*/'
    js_string = re.sub(pattern, '', js_string, flags=re.MULTILINE|re.DOTALL)
    return js_string

def remove_comments_python(python_code):
    pattern = r'#.*?$|\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"'
    python_code = re.sub(pattern, '', python_code, flags=re.MULTILINE|re.DOTALL)
    return python_code

def remove_comments_php(php_code):
    php_code = remove_comments_javascript(php_code)
    php_code = remove_comments_python(php_code)
    return php_code

def remove_comments_ruby(ruby_code):
    ruby_code = re.sub(r'#.*', '', ruby_code)
    ruby_code = re.sub(r'=begin.*?=end', '', ruby_code, flags=re.DOTALL)
    return ruby_code

def remove_comments_java(java_code):
    pattern = r'//.*?$|/\*.*?\*/'
    java_code = re.sub(pattern, '', java_code, flags=re.MULTILINE|re.DOTALL)
    return java_code

def comment_pattern(language):
    if language == "Python":
        return r'#.*?$|\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"'
    elif language == "Java" or language == "JavaScript" or language == "PHP":
        return r'//.*?$|/\*.*?\*/'
    elif language == "Ruby":
        return r'=begin.*?=end'

def remove_comments(code, language):
    if language == "Python":
        remover = remove_comments_python
    elif language == "Java":
        remover = remove_comments_java
    elif language == "JavaScript":
        remover = remove_comments_javascript
    elif language == "PHP":
        remover = remove_comments_php
    elif language == "Ruby":
        remover = remove_comments_ruby
    else:
        print(f"Cant process language: {language}")
        return ""
    return remover(code)

def contextize_overlapping_comments_encoder(before, segment, after, language):
    if language == "Python":
        return before, segment, after
    pattern = comment_pattern(language)
    if not pattern:
        print(f"Cant process language: {language}")
        return "", "", ""
    combined = before + segment + after
    matches = re.finditer(pattern, combined, flags=re.MULTILINE|re.DOTALL)
    start_buggy_segment = len(before)
    end_buggy_segment = len(before) + len(segment)

    def start_of_line(offset):
        while combined[offset] != "\n" and offset > 0:
            offset -= 1
        return offset + 1
    
    for match in matches:
        start = match.start()
        end = match.end()
        if start <= start_buggy_segment <= end:
            if end == len(combined):
                end -= 1
            start_buggy_segment = start_of_line(end)
        elif start <= end_buggy_segment <= end:
            end_buggy_segment = start_of_line(start)
    before = combined[:start_buggy_segment]
    segment = combined[start_buggy_segment:end_buggy_segment]
    after = combined[end_buggy_segment:]
    return before, segment, after

def contextize_overlapping_comments_decoder(before, segment, language):
    if language == "Python":
        return before, segment
    pattern = comment_pattern(language)
    if not pattern:
        print(f"Cant process language: {language}")
        return "", "", ""
    combined = before + segment
    # test for unclosed comment
    if len(re.findall(pattern, combined, flags=re.MULTILINE|re.DOTALL)) < len(re.findall(pattern, combined + "*/", flags=re.MULTILINE|re.DOTALL)):
        segment += "*/\n"
        combined = before + segment
    matches = re.finditer(pattern, combined, flags=re.MULTILINE|re.DOTALL)
    start_buggy_segment = len(before)

    def start_of_line(offset):
        while combined[offset] != "\n" and offset > 0:
            offset -= 1
        return offset + 1
    
    for match in matches:
        start = match.start()
        end = match.end()
        if start <= start_buggy_segment <= end:
            if end == len(combined): # can happen if fix is empty
                end -= 1
            start_buggy_segment = start_of_line(end)

    before = combined[:start_buggy_segment]
    segment = combined[start_buggy_segment:]
    return before, segment


def remove_blank_lines(code):
    code = re.sub(r'^[ \t]+$', '', code, flags=re.MULTILINE)
    while code.find("\n\n") >= 0:
        code = code.replace("\n\n","\n")
    return code

js_code = """
// This is a comment
var x = 10;

/* This is a
   multi-line comment */
var y = 20; // Another comment

console.log('safasdf');
console.log("asdf");

console.log(x + y);
"""

python_code = """
# This is a comment
def greet(name):
    # This is another comment
    print("Hello, " + name)

greet('Alice')  # Comment after code
"""

java_code = '''
/* This is a comment */
public class HelloWorld {
    // This is another comment
    public static void main(String[] args) {
        System.out.println("Hello, World!"); // Comment after code
    }
.csv}
'''

def process_repo(data):
    path, nmr = data
    csv_in, csv_out = path
    # print(f"Start processing {csv_in}.")
    try:
        df = pd.read_csv(
            csv_in,
            names=[
                "encoder_context_until",
                "buggy_segment",
                "encoder_context_from",
                "decoder_context",
                "fixed_segment",
                "author",
                "repo_name",
                "commit_hash",
                "language",
                "commit_date",
                "watch_count",
                "hash_id",
                "irgendwas",
            ],
            dtype={
                "encoder_context_until": "string",
                "buggy_segment": "string",
                "encoder_context_from": "string",
                "decoder_context": "string",
                "fixed_segment": "string",
                "author": "string",
                "repo_name": "string",
                "commit_hash": "string",
                "language": "string",
                "commit_date": "string",
                "irgendwas": "string",
                "watch_count": "int",
                "hash_id": "string",
            },
        )
    except Exception as exception:
        print(exception)
        return

    df = df.replace(pd.NA, "")

    # make overlapping comments part of the context
    processed_values = df.apply(
        lambda row: contextize_overlapping_comments_encoder(
            row['encoder_context_until'],
            row['buggy_segment'],
            row['encoder_context_from'],
            row['language'],
        ),
        axis=1,
    )
    df['encoder_context_until'] = [value[0] for value in processed_values]
    df['buggy_segment'] = [value[1] for value in processed_values]
    df['encoder_context_from'] = [value[2] for value in processed_values]

    processed_values = df.apply(
        lambda row: contextize_overlapping_comments_decoder(
            row['decoder_context'],
            row['fixed_segment'],
            row['language'],
        ),
        axis=1,
    )
    df['decoder_context'] = [value[0] for value in processed_values]
    df['fixed_segment'] = [value[1] for value in processed_values]

    # remove comments
    df['buggy_segment'] = df.apply(lambda row: remove_comments(row['buggy_segment'], row['language']), axis=1)
    df['fixed_segment'] = df.apply(lambda row: remove_comments(row['fixed_segment'], row['language']), axis=1)


    # remove blank lines
    df['fixed_segment'] = df['fixed_segment'].apply(remove_blank_lines)
    df['buggy_segment'] = df['buggy_segment'].apply(remove_blank_lines)

    # delete entries with blank fix
    df = df.dropna(subset=["fixed_segment"])
    df['fixed_segment'] = df['fixed_segment'].astype(str)
    df["fix_stripped"] = df['fixed_segment'].str.strip()
    df = df[df['fix_stripped'] != '']
    df.drop("fix_stripped", axis=1, inplace=True)

    # if the DataFrame is empty further steps would throw an error
    if len(df) == 0:
        return

    # Drop duplicate entries based on the "hash" column
    df["hash"] = df.apply(check_it, axis=1)
    df.drop_duplicates(subset="hash", keep="first", inplace=True)
    df.drop("hash", axis=1, inplace=True)

    # if the DataFrame is empty further steps would throw an error
    if len(df) == 0:
        return

    # Drop unchanged fixes
    mask = df.apply(lambda row: is_different(row['buggy_segment'], row['fixed_segment']), axis=1)
    df = df[mask]

    df.drop("irgendwas", axis=1, inplace=True)
    df.to_csv(csv_out, index=False, header=False)
    # print(f"Done processing {nmr}th repo {csv_out}.")

if not os.path.isdir("samples"):
    os.mkdir("samples")

filenames = [
   (f"segment_pairs/{path}", f"samples/{path}")
   for path in os.listdir("segment_pairs")
]
print("Preprocessing segment pairs.")
count = [i for i in range(1, len(filenames) + 1)]
with Pool(processes=cpu_count()) as pool:
    with tqdm(total=len(filenames)) as pbar:
        for _ in pool.imap_unordered(process_repo, zip(filenames, count)):
            pbar.update()
#  pool.map(
#      process_repo,
#      zip(filenames, count),
#  )

# ruby_code_with_comments = """
# # This is a single-line comment
# puts "Hello, world!" # This is also a comment
# 
# =begin
# This is a multi-line comment
# puts "Inside comment block"
# =end
# 
# # Another type of single-line comment
# 
# puts "Code outside the comments" # this is  a comment aswell
# """
# print(remove_comments_ruby(ruby_code_with_comments))

# php_code = """
# <?php
# // This is a single-line comment
# echo "Hello, world!"; // This is also a comment
# 
# /*
# This is a multi-line comment
# echo "Inside comment block";
# */
# 
# echo "hallo"; # Another type of single-line comment
# ?>
# """
# print(remove_comments_php(php_code))

#process_repo(((
#    "../samples_encoder_segments/webgme_webgme-engine.csv", "../samples_encoder_segments_processed/webgme_webgme-engine.csv"
#), 1))

#java_code_before = """
#public class ExampleClass {
#    public static void main(String[] args) {
#        /* 
#            This is a multi-line comment
#"""
#
#java_code_segment = """
#            that spans over several lines
#            and provides additional information about the code
#        */
#        
#        // Code statements
#        int x = 5;
#        int y = 10;
#        int sum = x + y;
#        
#"""
#
#
#before, segment = contextize_overlapping_comments_decoder(java_code_before, java_code_segment, "Java")
#print("before:")
#print(before)
#print("segment:")
#print(segment)
