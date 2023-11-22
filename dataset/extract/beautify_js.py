import os
import csv
import jsbeautifier
from multiprocessing import Pool, cpu_count
import pandas as pd
import logging

# Specify the column names in your CSV file
buggy_column = "buggy"
fixed_column = "fixed"
language_column = "language"


# Function to beautify JavaScript code
def beautify_javascript(code):
    options = jsbeautifier.default_options()
    try:
        beautified_code = jsbeautifier.beautify(code, options)
    except Exception as exception:
        print(exception)
        print("Failed to parse.")
        return code
    return beautified_code




def beautify_csv_file(data):

    filenames, count = data
    input_file, output_file = filenames

    try:
        df = pd.read_csv(
            input_file,
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
            },
        )
    except Exception as exception:
        print(exception)
        print(f"delete {input_file}.")
        os.remove(input_file)
        return

    # Iterate over each row and beautify JavaScript code samples
    for _, row in df.iterrows():
        try:
            if row["language"] == "JavaScript":
                row[buggy_column] = beautify_javascript(row[buggy_column])
                row[fixed_column] = beautify_javascript(row[fixed_column])
        except Exception as exception:
            print(input_file)
            logging.exception(exception)
            exit(0)
            continue
    # Write the updated DataFrame to the output CSV file
    df.to_csv(output_file, index=False, header=False)


if not os.path.isdir("method_pairs_beautified"):
    os.mkdir("method_pairs_beautified")

filenames = [(f"method_pairs/{path}", f"method_pairs_beautified/{path}") for path in os.listdir("method_pairs")]
count_all = len(filenames)
count = [i for i in range(1, count_all + 1)]

print("Beautifying JavaScript method pairs.")
with Pool(processes=cpu_count()) as pool:
    pool.map(
        beautify_csv_file,
        zip(filenames, count),
    )
