"""Fetch exp082-infer kernel output (script 322385897) and check LB."""
import json, os, tempfile, re
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp082-infer"

# Get kernel info
print("=== Kernel info ===")
try:
    info = api.kernels_view(slug)
    print(f"  Title: {info.title}")
    print(f"  Last run: {info.last_run_time}")
    print(f"  Total votes: {info.total_votes}")
except Exception as e:
    print(f"  ERR: {e}")

# Get all submissions and find ones with exp082 reference
print("\n=== Recent submissions (look for exp082) ===")
try:
    subs = api.competition_submissions("birdclef-2026")
    for s in subs[:30]:
        date_str = str(s.date)[:19]
        score = getattr(s, 'public_score', '?')
        desc = getattr(s, 'description', '?') or ''
        file_name = getattr(s, 'file_name', '?')
        print(f"  {date_str}  score={score}  | desc='{desc[:60]}' | file={file_name}")
except Exception as e:
    print(f"  ERR: {e}")

# Get kernel output
print("\n=== Kernel output ===")
with tempfile.TemporaryDirectory() as td:
    try:
        api.kernels_output(slug, path=td, quiet=True)
        td_path = Path(td)
        for f in sorted(td_path.rglob("*")):
            if f.is_file():
                print(f"  {f.name}  {f.stat().st_size/1024:.1f}KB")
        log_files = list(td_path.rglob("*.log"))
        if log_files:
            content = log_files[0].read_text(encoding="utf-8", errors="replace")
            if '"stream_name"' in content:
                lines_to_print = []
                for m in re.finditer(r'"data":"([^"]*(?:\\.[^"]*)*)"', content):
                    s = m.group(1).encode().decode('unicode_escape').rstrip('\n')
                    if s.strip():
                        lines_to_print.append(s)
                # Print last 60 lines
                print("\nLast 60 log lines:")
                for ln in lines_to_print[-60:]:
                    print(f"  {ln}")
    except Exception as e:
        print(f"  ERR: {e}")
