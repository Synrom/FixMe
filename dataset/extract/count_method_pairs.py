import os
import pandas as pd

count = 0
for path in os.listdir("method_pairs"):
    df = pd.read_csv(
        f"method_pairs/{path}",
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
    count += len(df)

print(f"Got {count} method pairs")
