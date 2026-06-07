"""v5 patch for exp069b: Move all train tensors to DEVICE.

V4 で init_prototypes だけ修正したが、_train_one 内部で model(train_emb, ...) 呼び出し時に
train_emb/train_logits/train_site/train_hour/train_prior が CPU で device mismatch 再発。

Fix: 各 tensor 作成時 / forward 直前で .to(DEVICE) 追加
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    # Cell that creates train tensors (line 759-764)
    {
        "old": 'train_emb = torch.tensor(lab_emb_files, dtype=torch.float32)\n',
        "new": 'train_emb = torch.tensor(lab_emb_files, dtype=torch.float32).to(DEVICE)\n',
    },
    {
        "old": 'train_logits = torch.tensor(lab_scores_files, dtype=torch.float32)\n',
        "new": 'train_logits = torch.tensor(lab_scores_files, dtype=torch.float32).to(DEVICE)\n',
    },
    {
        "old": 'train_site = torch.tensor(lab_site_ids, dtype=torch.long)\n',
        "new": 'train_site = torch.tensor(lab_site_ids, dtype=torch.long).to(DEVICE)\n',
    },
    {
        "old": 'train_hour = torch.tensor(lab_hours, dtype=torch.long)\n',
        "new": 'train_hour = torch.tensor(lab_hours, dtype=torch.long).to(DEVICE)\n',
    },
    {
        "old": 'train_prior = torch.tensor(lab_prior_files, dtype=torch.float32)\n',
        "new": 'train_prior = torch.tensor(lab_prior_files, dtype=torch.float32).to(DEVICE)\n',
    },
    # teacher_prob is derived from train_logits, will follow device automatically
    # but lab_flat / emb_flat used for init_prototypes were already fixed in v4
]

total = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    new_src = src
    for p in patches:
        if p["old"] in new_src:
            new_src = new_src.replace(p["old"], p["new"], 1)
            total += 1
            print(f"  Cell {i}: patched {p['old'][:60].strip()}")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

print(f"\nTotal patches: {total}/{len(patches)}")
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v5 patched: {NB_PATH}")
