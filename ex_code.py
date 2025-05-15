""" ruri-v3-310mが使えるかどうか確認するためのコード """
import time

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

start_time = time.time()

# Download from the 🤗 Hub
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("./models/ruri-v3-310m", device=device)

# Ruri v3 employs a 1+3 prefix scheme to distinguish between different types of text inputs:
# "" (empty string) is used for encoding semantic meaning.
# "トピック: " is used for classification, clustering, and encoding topical information.
# "検索クエリ: " is used for queries in retrieval tasks.
# "検索文書: " is used for documents to be retrieved.
sentences = [
    "川べりでサーフボードを持った人たちがいます",
    "サーファーたちが川べりに立っています",
    "トピック: 瑠璃色のサーファー",
    "検索クエリ: 瑠璃色はどんな色？",
    "検索文書: 瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。",
]

embeddings = model.encode(sentences, convert_to_tensor=True)
print(embeddings.size())
# [5, 768]

similarities = F.cosine_similarity(
    embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2
)
print(similarities)
# [[1.0000, 0.9603, 0.8157, 0.7074, 0.6916],
#  [0.9603, 1.0000, 0.8192, 0.7014, 0.6819],
#  [0.8157, 0.8192, 1.0000, 0.8701, 0.8470],
#  [0.7074, 0.7014, 0.8701, 1.0000, 0.9746],
#  [0.6916, 0.6819, 0.8470, 0.9746, 1.0000]]

end_time = time.time()  # 実行終了時刻を記録
print(f"実行時間: {end_time - start_time:.2f}秒") # 経過時間を表示
