import os
import pandas as pd
from multiprocessing import Pool, current_process, Manager, cpu_count
from tqdm import tqdm


def process_repo(data):
    path, nmr = data
    #print(f"Start processing repo {path}.")
    df = pd.read_csv(
        f"samples/{path}",
        names = [
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
            "watch_count": "int",
        },
    )
    df["commit_date"] = pd.to_datetime(df["commit_date"])
    df = df.sort_values("commit_date", ascending=False)
    split_index = int(len(df) * 0.1)
    testing_set = df.head(split_index)
    training_set = df.tail(len(df) - split_index)
    validation_set = training_set.sample(frac=0.1 / 0.9, random_state=42)
    training_set = training_set.drop(validation_set.index)
    testing_set.to_csv(f"testing/{path}", index=False, header=False)
    training_set.to_csv(f"training/{path}", index=False, header=False)
    validation_set.to_csv(
        f"validation/{path}", index=False, header=False
    )
    #print(f"Done processing repo {path} nmbr {nmr}.")

if not os.path.isdir("testing"):
    os.mkdir("testing")
if not os.path.isdir("training"):
    os.mkdir("training")
if not os.path.isdir("validation"):
    os.mkdir("validation")

filenames = [
    path for path in os.listdir("samples")
]
count = [i for i in range(1, len(filenames) + 1)]
with Pool(processes=cpu_count()) as pool:
    with tqdm(total=len(filenames)) as pbar:
        for _ in pool.imap_unordered(process_repo, zip(filenames, count)):
            pbar.update()
#    pool.map(
#        process_repo,
#        zip(filenames, count),
#    )
