"""Split kb_clean.jsonl into N batches for parallel sub-agent translation judging."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
src = OUT / "kb_clean.jsonl"
batches_dir = OUT / "judge_batches"
batches_dir.mkdir(exist_ok=True)

pairs = []
with open(src, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("description_en") and obj.get("description_sk"):
            pairs.append(obj)

N = 8
batch_size = (len(pairs) + N - 1) // N
print(f"Total pairs: {len(pairs)}; making {N} batches of ~{batch_size}")

for i in range(N):
    chunk = pairs[i * batch_size : (i + 1) * batch_size]
    if not chunk:
        continue
    out_path = batches_dir / f"batch_{i:02d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in chunk:
            f.write(json.dumps({
                "knowledgeId": obj["knowledgeId"],
                "description_sk": obj["description_sk"],
                "description_en": obj["description_en"],
            }, ensure_ascii=False) + "\n")
    print(f"  {out_path.name}: {len(chunk)} pairs")
