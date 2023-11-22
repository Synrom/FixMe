# FixMe
- defects4j must be installed on the system
- bugs2fix src2abs must be installed on the system

# Dataset

- the dataset can be downloaded at {url}

## Dataset Extraction
[phply](https://github.com/viraptor/phply) needs to be downloaded and places in `dataset/extract/`.\
To extract the dataset on your own run:

```bash
$ cd dataset
$ echo "github_token=my_github_token" > .env
$ pip install -r extract/requirements.txt
$ ./extract/create_dataset.sh <repository csv path> <commit csv path>
```

The repository CSV path can be `high_watchcount_repositories.csv` for the high watch count repositories and the commits CSV path can be `high_watch_count_commits.csv` for the high watch count commits. \
Similarly, the paths would be `top_200_repositories.csv` and `top_200_commits.csv` for the default dataset.


# Defects4j

- first execute `initialize.sh`. That creates the directories `repo_dir` and `bugids_dir` and `bugs2fix_dir`
- the execute `create.py` to extract a csv of bug samples for our models
- `replace_and_test.py` inserts predicted fixes in repositores and executes test suites
- `bugs2fix/create.py` creates a csv of samples of abstracted bugs for models trained on bugs2fix
- `bugs2fix/abs2src.py` maps abstract fix predictions back to their concrete form
- `bugs2fix/replace_and_test.py` inserts predicted fixes in repositores and executes test suites

