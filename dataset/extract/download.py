import logging
from github import Github
from git import Repo
import pandas as pd
from pydantic import BaseSettings
import os
import sys


class Settings(BaseSettings):
    github_token: str

    class Config:
        env_file = ".env"


settings = Settings()


def download_repository(author: str, repo: str, access_token: str) -> str:
    g = Github(access_token)

    # Retrieve the repository object
    try:
        repository = g.get_repo(f"{author}/{repo}")
    except Exception as exception:
        logging.exception(exception)
        logging.error(f"Got exception while cloning {author}/{repo}")
        return ""

    # Clone the repository using GitPython or perform any other desired actions
    clone_url = repository.clone_url

    path = f"repositories/{author}/{repo}"
    try:
        Repo.clone_from(
            clone_url, path, allow_unsafe_options=True, allow_unsafe_protocols=True
        )
    except Exception as exception:
        logging.exception(exception)
        logging.error(f"Got exception while cloning {author}/{repo}")
        return ""
    logging.info("Done.")

    return path

repository_path = sys.argv[1]
logging.info(f"Downloading repositories listed in {repository_path}.")

repos = pd.read_csv(
    repository_path, dtype={"name": "string", "language": "string", "commits": "int"}
)
logging.basicConfig(level=logging.INFO)

for index, row in repos.iterrows():
    author, repo = row["name"].split("/")
    logging.info(f"Downloding {author}/{repo} ...")
    download_repository(author, repo, settings.github_token)
