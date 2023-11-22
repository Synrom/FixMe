import pandas as pd

# Read the first CSV file into a DataFrame
def create_java_only(path1, path2):
    df = pd.read_csv(
        path1,
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
        },
    )
    print(f"{path1} has {len(df)} samples")
    df = df[df["language"] == "Java"]
    df.to_csv(path2, index=False, header=False)
    print(f"{path2} has {len(df)} samples")
    


# Read the first CSV file into a DataFrame
create_java_only("train.csv", "train_java.csv")
create_java_only("test.csv", "test_java.csv")
create_java_only("valid.csv", "valid_java.csv")
