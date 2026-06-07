"""Generate Kaggle NB: nb_xc_audio_dl.ipynb

Polite XC audio download from yasunorim/xc-birdclef-2026-target-urls.
- Rate limiting (1.5s delay/request)
- User-Agent header
- Exponential backoff retry
- Resume capability (skip existing files)
- Storage monitoring (stop at 18 GB safety margin)
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_audio_dl.ipynb")

def code_cell(cid, src):
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "source": src.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }

def md_cell(cid, src):
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


nb = {
    "cells": [
        md_cell("hdr", """# BC2026 XC Audio Downloader

**Polite Xeno-Canto audio downloader** for BC2026 Aves target species pretrain.

## Input
- `yasunorim/xc-birdclef-2026-target-urls` (CSV with 159 Aves species URLs)

## Output
- `/kaggle/working/xc_audio/<species>/XC<id>.mp3` (audio files)
- `/kaggle/working/xc_audio/_state.csv` (DL progress tracker for resume)
- `/kaggle/working/xc_audio/_metadata.csv` (final metadata + DL status)

## Politeness measures

| Setting | Value | Reason |
|---|---|---|
| Delay between requests | **1.5 sec** | XC server load |
| Retry on fail | 3 attempts | with exponential backoff [5, 15, 60] sec |
| User-Agent | identifying | XC ToS / netiquette |
| Resume | enabled | 9h session limit, can restart |
| Storage cap | 18 GB | /kaggle/working 20 GB safety margin |

## Expected
- 5,000-10,000 recordings (159 species × ~30-60 avg)
- 5-15 GB total
- 3-9h DL time (1 session may be enough; if not, restart and resume)

## After completion
1. Verify metadata
2. Save as Kaggle Dataset (`birdclef2026-xc-aves-audio`)
3. Use as input for exp047 pretrain NB
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup
# ============================================================
import os, time, json, sys, traceback
from pathlib import Path
import pandas as pd
import requests
from tqdm.auto import tqdm

# Config
INPUT_DIR = Path("/kaggle/input/xc-birdclef-2026-target-urls")
OUT_DIR = Path("/kaggle/working/xc_audio")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Politeness config (重要、これらは触らない)
REQUEST_DELAY_SEC = 1.5        # ★ XC server に優しく、1.5s/req = ~2400 req/hour
REQUEST_TIMEOUT = 30           # per-request timeout
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = [5, 15, 60]  # exponential backoff per retry
USER_AGENT = (
    "BirdCLEF2026-research/1.0 "
    "(Kaggle research project; contact: kaggle.com/maekeso)"
)

# Filters (already applied in yasunorim, but double-check)
QUALITY_FILTER = ["A", "B"]    # high quality only
MAX_LENGTH_SEC = 120           # skip >2 min (long file = large I/O + storage)
MAX_PER_SPECIES = None         # None = all; set int to limit if storage tight

# Safety
STORAGE_CAP_GB = 18.0          # stop early at 18 GB (working dir 20 GB limit)
MAX_RUNTIME_HOURS = 8.5        # stop early at 8.5h (9h Kaggle timeout)

print(f"Kaggle working dir: {OUT_DIR}")
print(f"Polite config:")
print(f"  delay/req: {REQUEST_DELAY_SEC}s")
print(f"  max retries: {MAX_RETRIES} with backoff {RETRY_BACKOFF_SEC}s")
print(f"  User-Agent: {USER_AGENT}")
print(f"Storage cap: {STORAGE_CAP_GB} GB")
print(f"Runtime cap: {MAX_RUNTIME_HOURS}h")
"""),

        code_cell("load_csv", """# ============================================================
# Cell 2: Load + Filter URL list
# ============================================================
csv_path = INPUT_DIR / "xc_filtered.csv"
assert csv_path.exists(), f"CSV not found: {csv_path}"
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} URLs")
print(f"Columns: {df.columns.tolist()}")
print(f"\\nSample row:")
print(df.iloc[0].to_dict())

# Apply filters
n_before = len(df)
if "quality" in df.columns and QUALITY_FILTER:
    df = df[df["quality"].isin(QUALITY_FILTER)].reset_index(drop=True)
    print(f"\\n[filter quality {QUALITY_FILTER}] {n_before} -> {len(df)}")

if "length_sec" in df.columns and MAX_LENGTH_SEC:
    n_before = len(df)
    df = df[df["length_sec"] <= MAX_LENGTH_SEC].reset_index(drop=True)
    print(f"[filter length <={MAX_LENGTH_SEC}s] {n_before} -> {len(df)}")

if MAX_PER_SPECIES:
    n_before = len(df)
    df = df.groupby("scientific_name", group_keys=False).head(MAX_PER_SPECIES).reset_index(drop=True)
    print(f"[per-species cap {MAX_PER_SPECIES}] {n_before} -> {len(df)}")

# Show distribution
print(f"\\nFinal: {len(df)} URLs across {df['scientific_name'].nunique()} species")
print(f"Per-species recordings (top 10):")
print(df.groupby('scientific_name').size().sort_values(ascending=False).head(10))
print(f"\\nPer-species recordings (bottom 10):")
print(df.groupby('scientific_name').size().sort_values().head(10))
"""),

        code_cell("paths", """# ============================================================
# Cell 3: Build save paths + resume detection
# ============================================================
def safe_species_dir(name):
    \"\"\"Sanitize species name for filesystem.\"\"\"
    return str(name).replace(" ", "_").replace("/", "_").replace("'", "")

df["species_dir"] = df["scientific_name"].apply(safe_species_dir)
df["save_path"] = df.apply(
    lambda r: OUT_DIR / r["species_dir"] / f"XC{r['xc_id']}.mp3",
    axis=1,
)

# Check existing files (resume support)
df["already_dl"] = df["save_path"].apply(lambda p: p.exists() and p.stat().st_size > 1000)
n_existing = int(df["already_dl"].sum())
n_to_dl = len(df) - n_existing
print(f"Already downloaded: {n_existing}/{len(df)}")
print(f"To download: {n_to_dl}")

if n_existing > 0:
    sample_existing = df[df["already_dl"]].iloc[0]
    print(f"\\nExample existing file: {sample_existing['save_path']}")
    print(f"  size: {sample_existing['save_path'].stat().st_size/1e6:.2f} MB")
"""),

        code_cell("download", """# ============================================================
# Cell 4: Polite download loop with retry + resume + safety limits
# ============================================================
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

stats = {
    "ok": 0,
    "skip_existing": int(df["already_dl"].sum()),
    "fail_total": 0,
    "fail_4xx": 0,
    "fail_5xx": 0,
    "fail_timeout": 0,
    "fail_other": 0,
    "bytes_dl_session": 0,
    "fail_urls": [],
}

start_t = time.time()
to_dl = df[~df["already_dl"]].reset_index(drop=True)
print(f"Starting download of {len(to_dl)} files...")
print(f"Estimated time: {len(to_dl) * REQUEST_DELAY_SEC / 3600:.1f}h (at {REQUEST_DELAY_SEC}s/req)")

# Progress save interval
SAVE_EVERY = 200

# Periodic state save
def save_state():
    state_df = df[["xc_id", "scientific_name", "file_url", "save_path", "already_dl"]].copy()
    state_df["save_path"] = state_df["save_path"].astype(str)
    state_df["downloaded_now"] = state_df["save_path"].apply(lambda p: Path(p).exists() and Path(p).stat().st_size > 1000)
    state_df.to_csv(OUT_DIR / "_state.csv", index=False)

try:
    for idx, row in tqdm(to_dl.iterrows(), total=len(to_dl), desc="DL"):
        # Safety checks
        elapsed_h = (time.time() - start_t) / 3600
        if elapsed_h > MAX_RUNTIME_HOURS:
            print(f"\\n⚠ Hit runtime cap {MAX_RUNTIME_HOURS}h, stopping early. Resume in next session.")
            break

        # Storage check (poll every 50 files for performance)
        if (idx + 1) % 50 == 0:
            try:
                total_gb = sum(f.stat().st_size for f in OUT_DIR.rglob("*.mp3")) / 1e9
                if total_gb > STORAGE_CAP_GB:
                    print(f"\\n⚠ Storage cap reached ({total_gb:.2f} GB > {STORAGE_CAP_GB} GB), stopping.")
                    break
            except Exception:
                pass

        save_path = Path(row["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        url = row["file_url"]

        # Retry loop
        success = False
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
                if r.status_code == 200:
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    sz = save_path.stat().st_size
                    if sz < 1000:
                        save_path.unlink()
                        last_err = f"file too small ({sz}B)"
                        raise ValueError(last_err)
                    stats["ok"] += 1
                    stats["bytes_dl_session"] += sz
                    success = True
                    break
                elif 400 <= r.status_code < 500:
                    last_err = f"HTTP {r.status_code}"
                    stats["fail_4xx"] += 1
                    break  # 4xx は retry しない (URL 不正 or license 変更等)
                elif 500 <= r.status_code < 600:
                    last_err = f"HTTP {r.status_code}"
                    stats["fail_5xx"] += 1
                    # 5xx は retry
            except requests.Timeout:
                last_err = "timeout"
                stats["fail_timeout"] += 1
            except Exception as e:
                last_err = str(e)[:100]
                stats["fail_other"] += 1

            if not success and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SEC[min(attempt, len(RETRY_BACKOFF_SEC) - 1)]
                time.sleep(wait)

        if not success:
            stats["fail_total"] += 1
            stats["fail_urls"].append((url, last_err))
            if save_path.exists():
                try: save_path.unlink()
                except: pass

        # Polite delay between requests
        time.sleep(REQUEST_DELAY_SEC)

        # Periodic progress log + state save
        if (idx + 1) % SAVE_EVERY == 0:
            elapsed = time.time() - start_t
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remain_s = (len(to_dl) - idx - 1) / rate if rate > 0 else 0
            total_gb = stats["bytes_dl_session"] / 1e9
            print(f"  [{idx+1}/{len(to_dl)}] ok={stats['ok']} fail={stats['fail_total']} "
                  f"sess_GB={total_gb:.2f} rate={rate*60:.1f}/min ETA={remain_s/3600:.1f}h")
            save_state()

except KeyboardInterrupt:
    print("\\n⚠ Interrupted by user")
except Exception as e:
    print(f"\\n⚠ Unexpected error: {e}")
    traceback.print_exc()

# Final state save
save_state()

print(f"\\n{'='*60}\\nDownload session summary\\n{'='*60}")
print(f"  OK:            {stats['ok']}")
print(f"  Skip existing: {stats['skip_existing']}")
print(f"  Failed (4xx):  {stats['fail_4xx']}")
print(f"  Failed (5xx):  {stats['fail_5xx']}")
print(f"  Timeout:       {stats['fail_timeout']}")
print(f"  Other fail:    {stats['fail_other']}")
print(f"  Total fail:    {stats['fail_total']}")
print(f"  Session bytes: {stats['bytes_dl_session']/1e9:.2f} GB")
print(f"  Session time:  {(time.time()-start_t)/60:.1f} min")
"""),

        code_cell("verify", """# ============================================================
# Cell 5: Verify + final metadata
# ============================================================
# Re-scan all files
df["save_path_str"] = df["save_path"].astype(str)
df["downloaded"] = df["save_path"].apply(lambda p: Path(p).exists() and Path(p).stat().st_size > 1000)
df["file_size_mb"] = df["save_path"].apply(lambda p: Path(p).stat().st_size / 1e6 if Path(p).exists() else 0.0)

n_dl = int(df["downloaded"].sum())
total_gb = df["file_size_mb"].sum() / 1024
print(f"=== Final state ===")
print(f"  Total URLs: {len(df)}")
print(f"  Downloaded: {n_dl} ({100*n_dl/len(df):.1f}%)")
print(f"  Total size: {total_gb:.2f} GB")
print(f"  Avg per file: {df[df['downloaded']]['file_size_mb'].mean():.2f} MB")

# Per-species summary
sp_summary = df.groupby("scientific_name").agg(
    total_urls=("xc_id", "count"),
    downloaded=("downloaded", "sum"),
    total_mb=("file_size_mb", "sum"),
).reset_index()
sp_summary["completion_pct"] = 100 * sp_summary["downloaded"] / sp_summary["total_urls"]
sp_summary = sp_summary.sort_values("completion_pct", ascending=False)

print(f"\\n=== Per-species coverage (top 5) ===")
print(sp_summary.head().to_string(index=False))
print(f"\\n=== Per-species coverage (bottom 5) ===")
print(sp_summary.tail().to_string(index=False))

print(f"\\n=== Species coverage ===")
fully = (sp_summary["completion_pct"] == 100).sum()
partial = ((sp_summary["completion_pct"] < 100) & (sp_summary["completion_pct"] > 0)).sum()
none = (sp_summary["completion_pct"] == 0).sum()
print(f"  Fully downloaded: {fully}/{len(sp_summary)} species")
print(f"  Partial: {partial}")
print(f"  Not started: {none}")

# Save final metadata
final_meta = df[df["downloaded"]][["xc_id", "scientific_name", "save_path_str", "file_size_mb"]].copy()
final_meta.columns = ["xc_id", "scientific_name", "filename", "file_size_mb"]
final_meta["filename"] = final_meta["filename"].apply(lambda p: str(Path(p).relative_to(OUT_DIR)))
final_meta.to_csv(OUT_DIR / "_metadata.csv", index=False)
sp_summary.to_csv(OUT_DIR / "_species_summary.csv", index=False)

print(f"\\nSaved: {OUT_DIR / '_metadata.csv'}")
print(f"Saved: {OUT_DIR / '_species_summary.csv'}")
"""),

        code_cell("save_dataset", """# ============================================================
# Cell 6: (Optional) Save as Kaggle Dataset
# ============================================================
# After this NB completes (or in a separate NB), upload to Kaggle Dataset
# via: api.dataset_create_new() or kernel output → save as dataset
#
# Direct from this NB:
import json, shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate (Kaggle env has kaggle.json auto-loaded)
api = KaggleApi(); api.authenticate()

SLUG = "birdclef2026-xc-aves-audio"
TITLE = "BirdCLEF2026 XC Aves Audio"
USER = "maekeso"

DRY_RUN = True  # Set False to actually upload

if not DRY_RUN:
    # Prepare dataset-metadata.json
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],  # XC default license
    }
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Upload (try version up first, fallback to create new)
    try:
        api.dataset_create_version(folder=str(OUT_DIR), version_notes="initial",
                                    dir_mode="zip", quiet=False)
        print("OK new version uploaded")
    except Exception:
        try:
            api.dataset_create_new(folder=str(OUT_DIR), public=False,
                                    dir_mode="zip", quiet=False)
            print("OK new dataset created")
        except Exception as e:
            print(f"upload err: {str(e)[:300]}")
    print(f"URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
else:
    print("DRY_RUN=True, skipping upload")
    print(f"To upload: set DRY_RUN=False and re-run this cell")
    print(f"Will upload {n_dl} files ({total_gb:.2f} GB) to {USER}/{SLUG}")
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n_lines = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id', '?'):20s} L={n_lines}")
