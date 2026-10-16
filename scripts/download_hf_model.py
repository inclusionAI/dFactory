import argparse

from huggingface_hub import snapshot_download


def main(repo_id: str, local_dir: str) -> None:
    """Download one Hub repository into the exact requested directory."""
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()
    main(args.repo_id, args.local_dir)
