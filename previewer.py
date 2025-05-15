"""実際にファイルがどんなふうに動くかを確認するためのスクリプト。実際にはファイルの移動などは起こらない。"""

import argparse
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(
        description="Preview sorting of markdown files by folder embeddings"
    )
    parser.add_argument("root_path", help="Root directory path to scan")
    args = parser.parse_args()

    root = args.root_path.rstrip("/")

    # 設定読み込み
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, encoding="utf-8") as cf:
        config = yaml.safe_load(cf)
    exclude_dirs = config.get("exclude_dirs", [])
    include_exts = config.get("include_exts", [])

    # 除外ディレクトリ名を正規化（末尾スラッシュを削除）
    exclude_dirs = [d.rstrip("/\\") for d in exclude_dirs]

    # 1. サブディレクトリ収集（除外ディレクトリ以下は探索しない）
    folder_tuples = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for d in dirnames:
            folder_tuples.append((os.path.join(dirpath, d), d))
    if not folder_tuples:
        print("No subdirectories found under the specified root.")
        return
    folder_paths, folder_names = zip(*folder_tuples)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("./models/ruri-v3-310m", device=device)

    # Encode folder names into embeddings
    folder_embs = model.encode(folder_names, convert_to_tensor=True)

    # 2. ファイル取得（拡張子と除外ディレクトリを両方チェック）
    md_files = [
        f
        for f in glob.glob(f"{root}/**/*", recursive=True)
        if Path(f).suffix in include_exts
        and not any(part in exclude_dirs for part in Path(f).parts)
    ]
    if not md_files:
        print("No markdown files found under the specified root.")
        return

    print("Preview of file -> predicted folder:")
    # Compute embeddings for each markdown file and find the closest folder
    for filepath in md_files:
        with open(filepath, encoding="utf-8") as fp:
            text = fp.read()
        emb = model.encode(text, convert_to_tensor=True)
        sims = F.cosine_similarity(emb.unsqueeze(0), folder_embs, dim=1)
        idx = torch.argmax(sims).item()
        print(f"{filepath} -> {folder_paths[idx]}")


if __name__ == "__main__":
    main()
