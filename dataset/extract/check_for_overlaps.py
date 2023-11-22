import pandas as pd

# Read the first CSV file into a DataFrame
def check_overlaps(path1, path2):
    df1 = pd.read_csv(
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

    df2 = pd.read_csv(
        path2,
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

    # Extract the common column(s) from both DataFrames
    common_columns = [
        "encoder_context_until",
        "buggy_segment",
        "encoder_context_from",
        "decoder_context",
        "fixed_segment",
    ]  # Replace with the actual column names

    #df1 = pd.merge(df1, df2[common_columns], on=common_columns, how='left', indicator=True)
    #df1 = df1[df1['_merge'] == 'left_only']
    #df1 = df1.drop('_merge', axis=1)
    #df1.to_csv(path1, index=False, header=False)


    # Merge the two DataFrames based on the common column(s)
    merged = pd.merge(df1[common_columns], df2[common_columns], how='inner')

    # Check if there are any overlapping values
    if merged.empty:
        print(f"There are no overlaps between {path1} and {path2}.")
    else:
        print(f"There are {len(merged)} overlaps between {path1} and {path2}.")


# Read the first CSV file into a DataFrame
def delete_overlaps(path1, path2):
    df1 = pd.read_csv(
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

    df2 = pd.read_csv(
        path2,
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

    # Extract the common column(s) from both DataFrames
    common_columns = [
        "encoder_context_until",
        "buggy_segment",
        "encoder_context_from",
        "decoder_context",
        "fixed_segment",
    ]  # Replace with the actual column names

    df1 = pd.merge(df1, df2[common_columns], on=common_columns, how='left', indicator=True)
    df1 = df1[df1['_merge'] == 'left_only']
    df1 = df1.drop('_merge', axis=1)
    df1.to_csv(path1, index=False, header=False)


    # Merge the two DataFrames based on the common column(s)
    # merged = pd.merge(df1[common_columns], df2[common_columns], how='inner')

    # Check if there are any overlapping values
    # if merged.empty:
    #     print(f"There are no overlaps between {path1} and {path2}.")
    # else:
    #     print(f"There are {len(merged)} overlaps between {path1} and {path2}.")

training_filename = "train.csv"
testing_filename = "test.csv"
validation_filename = "valid.csv"
delete_overlaps(training_filename, validation_filename)
delete_overlaps(training_filename, testing_filename)
delete_overlaps(validation_filename, testing_filename)
