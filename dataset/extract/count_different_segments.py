import pandas as pd
import os
def process_repo(path):
    df = pd.read_csv(
        path,
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
    filtered = len(df[df["encoder_context_until"] != df["decoder_context"]])
    unfiltered = len(df)

    return filtered, unfiltered

filenames = [
   f"segment_pairs/{path}" for path in os.listdir("segment_pairs")
]
unfiltered = 0
filtered = 0
for filename in filenames:
    n1, n2 = process_repo(filename)
    unfiltered += n2
    filtered += n1
print(f"{filtered} out of {unfiltered} contain changes between encoder_context_until and decoder_context.")
