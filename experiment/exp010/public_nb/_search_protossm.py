import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

def show(rows, label):
    print(f"\n=== {label} ({len(rows)} hits) ===")
    for k in rows:
        score = getattr(k, "totalScore", None) or getattr(k, "scoreNullable", None) or "-"
        ref = getattr(k, "ref", "")
        title = getattr(k, "title", "")
        print(f"  {str(score):>8} | {ref:60s} | {title}")

# 1. protossm keyword
rows = api.kernels_list(competition="birdclef-2026", search="protossm",
                        sort_by="scoreDescending", page_size=30)
show(rows, "search='protossm'")

# 2. perch protossm
rows = api.kernels_list(competition="birdclef-2026", search="perch protossm",
                        sort_by="scoreDescending", page_size=30)
show(rows, "search='perch protossm'")

# 3. perch (broader)
rows = api.kernels_list(competition="birdclef-2026", search="perch",
                        sort_by="scoreDescending", page_size=30)
show(rows, "search='perch' top 30")

# 4. ssm
rows = api.kernels_list(competition="birdclef-2026", search="ssm",
                        sort_by="scoreDescending", page_size=20)
show(rows, "search='ssm' top 20")
