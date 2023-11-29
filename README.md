# Requirements

- defects4j must be installed on the system
- bugs2fix src2abs must be installed on the system

# Dataset

- the dataset can be downloaded at [anonymised]()

## Dataset Extraction
[phply](https://github.com/viraptor/phply) needs to be downloaded and places in `dataset/extract/`\
To extract the dataset on your own run:

```bash
$ cd dataset
$ echo "github_token=my_github_token" > .env
$ pip install -r extract/requirements.txt
$ ./extract/create_dataset.sh <repository csv path> <commit csv path>
```

The repository CSV path can be `high_watchcount_repositories.csv` for the high watch count repositories and the commits CSV path can be `high_watch_count_commits.csv` for the high watch count commits. \
Similarly, the paths would be `top_200_repositories.csv` and `top_200_commits.csv` for the default dataset.

# Training

The training processes starts by tokenizing all needed datasets. This can be done using the following command:
```bash
$ python -m fixme.run tokenize <graphcodebert/unixcoder> <source CSV-path> <CSV-path to save tokenized dataset>
```

After that, a good learning rate can be figured out using a heuristic introduced by [Leslie N. Smith](https://ieeexplore.ieee.org/abstract/document/7926641)
```bash
$ python -m fixme.run learning <graphcodebert/unixcoder> <tokenized train path>
```

And finally, you can train your model:
```bash
$ python -m fixme.run train <graphcodebert/unixcoder> <tokenized train path> <tokenized valid path> <ouput directory> <learning rate>
```

# Evaluation

To evaluate a model on a test set, you first need to tokenize the test set as described before.
After that, the model can be tested via:
```bash
$ python -m fixme.run test <graphcodebert/unixcoder> <tokenized test set> <output directory> <path to loaded model>
```

## Defects4j

Part of our evaluation works on the [Defects4J](https://github.com/rjust/defects4j) benchmark. The workflow here is as follows:
- first execute `initialize.sh`. That creates the directories `repo_dir` and `bugids_dir` and `bugs2fix_dir`
- then execute `create.py` to extract a CSV-file of bug samples
- the CSV then needs to be tokenized and after that, fixes are generated using the following command:
```bash
$ python -m fixme.run test <graphcodebert/unixcoder> <tokenized test set> <output directory> <path to loaded model>
```
- `replace_and_test.py` inserts predicted fixes in repositores and executes test suites

## Bugs2Fix

Before the model can be trained on [Bugs2Fix](https://sites.google.com/view/learning-fixes), the dataset first must be tokenized. This can be done using the following command:
```bash
$ python -m fixme.run tokenize_bugs2fix  <graphcodebert/unixcoder> <source path>,<target path> <CSV-path to save tokenized dataset>
```
After that, the evaluation can be done using the following python scripts:
- `bugs2fix/create.py` creates a CSV-file containing samples of abstracted bugs
- `bugs2fix/abs2src.py` maps abstract fix predictions back to their concrete form
- `bugs2fix/replace_and_test.py` inserts predicted fixes in repositores and executes test suites

