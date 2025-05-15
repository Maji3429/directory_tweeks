"""実際にフォルダ名の埋め込みに基づいてファイルを自動で移動するスクリプト"""

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
        description="フォルダ名の埋め込みに基づいて.mdファイルを自動で移動するスクリプト"
    )
    parser.add_argument("root_path", help="対象となるルートディレクトリのパス")
    args = parser.parse_args()

    root = args.root_path.rstrip("/")

    # 設定読み込み
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, encoding="utf-8") as cf:
        config = yaml.safe_load(cf)
    exclude_dirs = set(config.get("exclude_dirs", []))
    include_exts = set(config.get("include_exts", []))

    # 除外ディレクトリ名を正規化
    exclude_dirs = {d.rstrip("/\\") for d in exclude_dirs}

    # 1. サブディレクトリ収集（除外ディレクトリ以下を探索しない）
    folder_tuples = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for d in dirnames:
            folder_tuples.append((os.path.join(dirpath, d), d))
    if not folder_tuples:
        print("⚠️ サブディレクトリが見つかりませんでした。")
        return

    folder_paths, folder_names = zip(*folder_tuples)

    # 2. 埋め込みモデルのロード
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("./models/ruri-v3-310m", device=device)

    # 3. フォルダ名をベクトル化
    #    必要なら "トピック: {name}" の形式に変更してもOK
    folder_embs = model.encode(list(folder_names), convert_to_tensor=True)

    # 4. ファイル取得（拡張子＆除外ディレクトリ両方チェック）
    md_files = [
        f
        for f in glob.glob(f"{root}/**/*", recursive=True)
        if Path(f).suffix in include_exts
        and not any(part in exclude_dirs for part in Path(f).parts)
    ]
    if not md_files:
        print("⚠️ Markdown ファイルが見つかりませんでした。")
        return

    # 5. 各ファイルを分類し、移動
    for filepath in md_files:
        text = open(filepath, encoding="utf-8").read()
        emb = model.encode(text, convert_to_tensor=True)
        sims = F.cosine_similarity(emb.unsqueeze(0), folder_embs, dim=1)
        idx = torch.argmax(sims).item()
        target_folder = folder_paths[idx]

        # すでに正しいフォルダ内ならスキップ
        current_dir = os.path.dirname(filepath)
        if os.path.abspath(current_dir) == os.path.abspath(target_folder):
            continue

        # 移動先ディレクトリを確保して移動（上書き）
        os.makedirs(target_folder, exist_ok=True)
        dest = os.path.join(target_folder, os.path.basename(filepath))
        os.replace(filepath, dest)
        print(f"Moved: {filepath} → {dest}")

    print("✅ ファイル移動が完了しました。")


if __name__ == "__main__":
    main()
