"""Generate Kaggle NB: nb_xc_api_dl.ipynb

XC API 直接 query で BC2026 234 species の追加録音を polite に DL。

Workflow:
  1. BC2026 taxonomy.csv 読込 (234 species)
  2. 各 species を XC API で query → URL list 構築
  3. Filter (quality, length, license) → DL
  4. Resume / safety limits 込で実行
  5. Kaggle Dataset 化
"""
import json
from pathlib import Path

OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl.ipynb")

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}

def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell("hdr", """# BC2026 XC API Direct Downloader

XC API を直接 query して **BC2026 234 species の追加録音**を polite に DL する。

## 流れ

1. BC2026 `taxonomy.csv` から 234 species 取得
2. 各 species を XC API で query (pagination 対応)
3. Filter (quality A/B/C、length ≤120s、license non-ND)
4. Polite DL (1.5s delay/req、retry 3 回、resume 対応)
5. Kaggle Dataset 化

## politeness

| Setting | Value | Reason |
|---|---|---|
| Delay (API query) | 1.5 sec | XC API rate limit 配慮 |
| Delay (audio DL) | 1.5 sec | XC server load |
| Retry | 3 attempts | exponential backoff [5,15,60]s |
| User-Agent | identifying | XC ToS / 礼儀 |
| Resume | enabled | 9h session 制限対応 |
| Storage cap | 18 GB | /kaggle/working 安全マージン |

## 期待規模

- 234 species × avg 50-200 recordings/species = 12k-50k 録音
- Quality A/B/C filter で ~10-30k
- 容量 ~10-30 GB

## 完了後

`maekeso/birdclef2026-xc-api-aves-audio` 等として Kaggle Dataset 化
"""),

        code_cell("setup", """# ============================================================
# Cell 1: Setup
# ============================================================
import os, time, json, sys
from pathlib import Path
import pandas as pd
import requests
from tqdm.auto import tqdm

# ★ XC API v3 (key 必須、v2 は deprecated)
# This NB is PRIVATE — key embedded directly. Keep NB private to protect key.
XC_API_KEY = "1eef1b2770d218e80cea18fc28621e8dcad01f6a"
XC_API_URL = "https://xeno-canto.org/api/3/recordings"

# Self-contained: species list 埋め込み (taxonomy.csv 不要)
OUT_DIR = Path("/kaggle/working/xc_api")
OUT_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR = OUT_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True, parents=True)
URL_CACHE = OUT_DIR / "_urls.csv"  # query result cache for resume

# Politeness config
QUERY_DELAY_SEC = 1.5        # XC API query 間隔
DL_DELAY_SEC = 1.5           # audio DL 間隔
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = [5, 15, 60]
USER_AGENT = (
    "BirdCLEF2026-research/1.0 "
    "(Kaggle research project; contact: kaggle.com/maekeso)"
)

# Filters
QUALITY_FILTER = ["A", "B", "C"]  # high-medium quality
MAX_LENGTH_SEC = 120              # ≤2 min only
EXCLUDE_ND_LICENSE = True         # No-Derivatives 除外 (再配布不可なため)
MAX_PER_SPECIES_QUERY = 500       # XC API 1 page = 500、複数 page 取得可
MAX_PER_SPECIES_DL = None         # None = all, set int to cap

# Safety
STORAGE_CAP_GB = 18.0
MAX_RUNTIME_HOURS = 8.5

print(f"Output: {OUT_DIR}")
print(f"Polite delays: query={QUERY_DELAY_SEC}s, dl={DL_DELAY_SEC}s")
print(f"Filters: quality={QUALITY_FILTER}, length<={MAX_LENGTH_SEC}s, exclude_ND={EXCLUDE_ND_LICENSE}")
"""),

        code_cell("aves_list", """# ============================================================
# Cell 2: BC2026 Aves species list (embedded, no external dependency)
# Auto-generated from BC2026 taxonomy.csv (class_name == 'Aves')
# 162 species
# ============================================================
__BC2026_AVES_RAW = [
    ('Hylophilus pectoralis', 'ashgre1'),
    ('Mustelirallus albicollis', 'astcra1'),
    ('Crax fasciolata', 'bafcur1'),
    ('Micrastur ruficollis', 'baffal1'),
    ('Coereba flaveola', 'banana'),
    ('Thamnophilus doliatus', 'barant1'),
    ('Procnias nudicollis', 'batbel1'),
    ('Ara ararauna', 'baymac'),
    ('Dendrocygna autumnalis', 'bbwduc'),
    ('Microspingus melanoleucus', 'bcwfin2'),
    ('Donacobius atricapilla', 'bkcdon'),
    ('Aratinga nenday', 'bkhpar'),
    ('Busarellus nigricollis', 'blchaw1'),
    ('Spizaetus tyrannus', 'blheag1'),
    ('Tityra cayana', 'blttit1'),
    ('Myiarchus tyrannulus', 'bncfly'),
    ('Megarynchus pitangua', 'bobfly1'),
    ('Progne tapera', 'brcmar1'),
    ('Tyto furcata', 'brnowl'),
    ('Momotus momota', 'bucmot4'),
    ('Thectocercus acuticaudatus', 'bucpar'),
    ('Amazona aestiva', 'bufpar'),
    ('Theristicus caudatus', 'bunibi1'),
    ('Athene cunicularia', 'burowl'),
    ('Colaptes campestris', 'camfli1'),
    ('Ortalis canicollis', 'chacha1'),
    ('Mimus saturninus', 'chbmoc1'),
    ('Gnorimopsar chopi', 'chobla1'),
    ('Conirostrum speciosum', 'chvcon1'),
    ('Synallaxis hypospodia', 'cibspi1'),
    ('Micrastur semitorquatus', 'coffal1'),
    ('Nyctidromus albicollis', 'compau'),
    ('Nyctibius griseus', 'compot1'),
    ('Turdus amaurochalinus', 'crbthr1'),
    ('Pachyramphus validus', 'crebec1'),
    ('Taoniscus nanus', 'dwatin1'),
    ('Icterus pyrrhopterus', 'epaori4'),
    ('Lathrotriccus euleri', 'eulfly1'),
    ('Cantorchilus guarayanus', 'fabwre1'),
    ('Glaucidium brasilianum', 'fepowl'),
    ('Machaeropterus pyrocephalus', 'ficman1'),
    ('Myiothlypis flaveola', 'flawar1'),
    ('Tyrannus savana', 'fotfly'),
    ('Cnemotriccus fuscatus', 'fusfly1'),
    ('Hylocharis chrysura', 'gilhum1'),
    ('Aramides ypecaha', 'giwrai1'),
    ('Chionomesa fimbriata', 'glteme1'),
    ('Saltator coerulescens', 'grasal3'),
    ('Crotophaga major', 'greani1'),
    ('Taraba major', 'greant1'),
    ('Myiopagis viridicata', 'greela'),
    ('Pitangus sulphuratus', 'grekis'),
    ('Nyctibius grandis', 'grepot1'),
    ('Phacellodomus ruber', 'gretho2'),
    ('Tringa melanoleuca', 'greyel'),
    ('Leptotila rufaxilla', 'grfdov1'),
    ('Eucometis penicillata', 'grhtan1'),
    ('Aramides cajaneus', 'gycwor1'),
    ('Anhima cornuta', 'horscr1'),
    ('Passer domesticus', 'houspa'),
    ('Anodorhynchus hyacinthinus', 'hyamac1'),
    ('Elaenia spectabilis', 'larela1'),
    ('Elaenia chiriquensis', 'lesela1'),
    ('Emberizoides ypiranganus', 'lesgrf1'),
    ('Aramus guarauna', 'limpki'),
    ('Dryocopus lineatus', 'linwoo1'),
    ('Coccycua minuta', 'litcuc2'),
    ('Setopagis parvula', 'litnig1'),
    ('Pyrrhura frontalis', 'mabpar'),
    ('Cercomacra melanaria', 'magant1'),
    ('Cissopis leverianus', 'magtan2'),
    ('Polioptila dumicola', 'masgna1'),
    ('Chordeiles nacunda', 'nacnig1'),
    ('Rufirallus schomburgkii', 'ocecra1'),
    ('Sittasomus griseicapillus', 'oliwoo1'),
    ('Icterus croconotus', 'orbtro3'),
    ('Amazona amazonica', 'orwpar'),
    ('Pandion haliaetus', 'osprey'),
    ('Synallaxis albescens', 'pabspi1'),
    ('Furnarius leucopus', 'palhor3'),
    ('Thraupis palmarum', 'paltan1'),
    ('Dromococcyx phasianellus', 'phecuc1'),
    ('Patagioenas picazuro', 'picpig2'),
    ('Legatus leucophaius', 'pirfly1'),
    ('Thamnophilus pelzelni', 'plasla1'),
    ('Inezia inornata', 'platyr1'),
    ('Cyanocorax chrysops', 'plcjay1'),
    ('Theristicus caerulescens', 'pluibi1'),
    ('Cyanocorax cyanomelas', 'purjay1'),
    ('Hemitriccus margaritaceiventer', 'pvttyr1'),
    ('Ara chloropterus', 'ragmac1'),
    ('Campylorhamphus trochilirostris', 'rebscy1'),
    ('Coryphospingus cucullatus', 'recfin1'),
    ('Gallus gallus', 'redjun'),
    ('Cariama cristata', 'relser1'),
    ('Megaceryle torquata', 'rinkin1'),
    ('Myiothlypis rivularis', 'rivwar1'),
    ('Rupornis magnirostris', 'roahaw'),
    ('Turdus rufiventris', 'rubthr1'),
    ('Pseudoseisura unirufa', 'rufcac2'),
    ('Casiornis rufus', 'rufcas2'),
    ('Conopophaga lineata', 'rufgna3'),
    ('Furnarius rufus', 'rufhor2'),
    ('Antrostomus rufus', 'rufnig1'),
    ('Phacellodomus rufifrons', 'ruftho1'),
    ('Poecilotriccus latirostris', 'ruftof1'),
    ('Myiozetetes cayanensis', 'rumfly1'),
    ('Tigrisoma lineatum', 'ruther1'),
    ('Galbula ruficauda', 'rutjac1'),
    ('Arremon flavirostris', 'sabspa1'),
    ('Sicalis flaveola', 'saffin'),
    ('Thraupis sayaca', 'saytan1'),
    ('Columbina squammata', 'scadov1'),
    ('Pionus maximiliani', 'schpar1'),
    ('Phaethornis eurynome', 'scther1'),
    ('Myiarchus ferox', 'shcfly1'),
    ('Accipiter striatus', 'shshaw'),
    ('Lurocalis semitorquatus', 'shtnig1'),
    ('Ramphocelus carbo', 'sibtan2'),
    ('Crotophaga ani', 'smbani'),
    ('Crypturellus parvirostris', 'smbtin1'),
    ('Cacicus solitarius', 'sobcac1'),
    ('Camptostoma obsoletum', 'sobtyr1'),
    ('Myiozetetes similis', 'socfly1'),
    ('Synallaxis frontalis', 'sofspi1'),
    ('Corythopis delalandi', 'souant1'),
    ('Vanellus chilensis', 'soulap1'),
    ('Chauna torquata', 'souscr1'),
    ('Hypoedaleus guttatus', 'spbant3'),
    ('Synallaxis spixi', 'spispi1'),
    ('Antiurus maculicaudus', 'sptnig1'),
    ('Piaya cayana', 'squcuc1'),
    ('Dendroplex picus', 'stbwoo2'),
    ('Tapera naevia', 'strcuc1'),
    ('Butorides striata', 'strher2'),
    ('Asio clamator', 'strowl1'),
    ('Eupetomena macroura', 'swthum1'),
    ('Chiroxiphia caudata', 'swtman1'),
    ('Crypturellus tataupa', 'tattin1'),
    ('Campylorhynchus turdinus', 'thlwre1'),
    ('Ramphastos toco', 'toctou1'),
    ('Tyrannus melancholicus', 'trokin'),
    ('Megascops choliba', 'trsowl'),
    ('Crypturellus undulatus', 'undtin1'),
    ('Thamnophilus caerulescens', 'varant1'),
    ('Jacana jacana', 'watjac1'),
    ('Pyriglena maura', 'wesfie1'),
    ('Dendrocygna viduata', 'wfwduc1'),
    ('Biatas nigropectus', 'whbant2'),
    ('Myiothlypis leucoblephara', 'whbwar2'),
    ('Melanerpes candidus', 'whiwoo1'),
    ('Synallaxis albilora', 'whlspi1'),
    ('Cyanocorax cyanopogon', 'whnjay1'),
    ('Leptotila verreauxi', 'whtdov'),
    ('Picumnus albosquamatus', 'whwpic1'),
    ('Caracara plancus', 'y00678'),
    ('Paroaria capitata', 'yebcar'),
    ('Elaenia flavogaster', 'yebela1'),
    ('Primolius auricollis', 'yecmac'),
    ('Brotogeris chiriri', 'yecpar'),
    ('Daptrius chimachima', 'yehcar1'),
    ('Tolmomyias sulphurescens', 'yeofly1'),
]

aves_df = pd.DataFrame(__BC2026_AVES_RAW, columns=["scientific_name", "primary_label"])
print(f"BC2026 Aves species (embedded): {len(aves_df)}")
print(aves_df.head().to_string(index=False))

# Parse genus + species
def parse_sci_name(name):
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return parts[0], ""

aves_df[["genus", "species"]] = aves_df["scientific_name"].apply(
    lambda x: pd.Series(parse_sci_name(x))
)
print(f"\\nUnique genera: {aves_df['genus'].nunique()}")
"""),

        code_cell("query_xc", """# ============================================================
# Cell 3: Query XC API for each species → build URL list (with cache)
# ============================================================
# XC API v3: https://xeno-canto.org/api/3/recordings?key=...&query=...
# Returns JSON with `recordings` list and pagination info (v2 was deprecated, 404)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

if URL_CACHE.exists():
    print(f"Loading cached URL list: {URL_CACHE}")
    url_df = pd.read_csv(URL_CACHE)
    print(f"  cached: {len(url_df)} URLs from {url_df['scientific_name'].nunique()} species")
    queried_species = set(url_df["scientific_name"].unique())
else:
    url_df = pd.DataFrame()
    queried_species = set()

records = [url_df] if len(url_df) else []
species_to_query = [s for s in aves_df["scientific_name"].tolist() if s not in queried_species]
print(f"\\nSpecies to query: {len(species_to_query)} (of {len(aves_df)})")

for idx, sci_name in enumerate(tqdm(species_to_query, desc="XC query")):
    genus, species = parse_sci_name(sci_name)
    species_records = []
    page = 1
    while True:
        # XC API v3 query (key required)
        q = f"gen:{genus} sp:{species}"
        params = {"query": q, "key": XC_API_KEY, "page": page}
        try:
            r = session.get(XC_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"\\n  [{sci_name}] page {page} query err: {str(e)[:150]}")
            break

        recs = data.get("recordings", [])
        if not recs:
            break

        for rec in recs:
            # v3 schema (likely same as v2 but some fields may differ)
            species_records.append({
                "scientific_name": sci_name,
                "xc_id": rec.get("id"),
                "genus": rec.get("gen"),
                "species": rec.get("sp"),
                "subspecies": rec.get("ssp"),
                "en_name": rec.get("en"),
                "type": rec.get("type"),
                "file_url": rec.get("file") or rec.get("file-name") or rec.get("download"),
                "quality": rec.get("q"),
                "length_str": rec.get("length"),
                "country": rec.get("cnt"),
                "license": rec.get("lic"),
                "rmk": rec.get("rmk"),
            })

        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1
        time.sleep(QUERY_DELAY_SEC)  # polite between pages

    records.append(pd.DataFrame(species_records))
    time.sleep(QUERY_DELAY_SEC)  # polite between species

    # Periodic save (every 20 species)
    if (idx + 1) % 20 == 0:
        url_df_partial = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        if len(url_df_partial):
            url_df_partial.to_csv(URL_CACHE, index=False)
            print(f"\\n  [{idx+1}/{len(species_to_query)}] saved {len(url_df_partial)} URLs to cache")

# Final concat + save
url_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
url_df.to_csv(URL_CACHE, index=False)

print(f"\\nFinal URL collection: {len(url_df)} URLs")
print(f"  from {url_df['scientific_name'].nunique()} species")
print(f"  quality dist: {url_df['quality'].value_counts().to_dict()}")
"""),

        code_cell("filter_urls", """# ============================================================
# Cell 4: Filter URLs (quality, length, license)
# ============================================================
def length_str_to_sec(s):
    \"\"\"XC length format 'MM:SS' or 'M:SS' → seconds.\"\"\"
    try:
        parts = str(s).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return None
    return None

url_df["length_sec"] = url_df["length_str"].apply(length_str_to_sec)
print(f"Total URLs: {len(url_df)}")

# Filter chain
n = len(url_df)
url_df = url_df.dropna(subset=["file_url", "xc_id"])
print(f"  after dropna(file_url/xc_id): {n} -> {len(url_df)}")

n = len(url_df)
url_df = url_df[url_df["quality"].isin(QUALITY_FILTER)].reset_index(drop=True)
print(f"  after quality filter {QUALITY_FILTER}: {n} -> {len(url_df)}")

n = len(url_df)
url_df = url_df[url_df["length_sec"].notna() & (url_df["length_sec"] <= MAX_LENGTH_SEC)].reset_index(drop=True)
print(f"  after length<={MAX_LENGTH_SEC}s: {n} -> {len(url_df)}")

if EXCLUDE_ND_LICENSE:
    n = len(url_df)
    # XC license typically contains "BY-NC-ND" or "BY-NC-SA" or "BY"
    is_nd = url_df["license"].astype(str).str.contains("nd", case=False, na=False)
    url_df = url_df[~is_nd].reset_index(drop=True)
    print(f"  after exclude ND license: {n} -> {len(url_df)}")

if MAX_PER_SPECIES_DL:
    n = len(url_df)
    url_df = url_df.groupby("scientific_name", group_keys=False).head(MAX_PER_SPECIES_DL).reset_index(drop=True)
    print(f"  after per-species cap {MAX_PER_SPECIES_DL}: {n} -> {len(url_df)}")

print(f"\\n=== After all filters: {len(url_df)} URLs ===")
print(f"  species covered: {url_df['scientific_name'].nunique()} / {len(aves_df)}")
print(f"  quality dist: {url_df['quality'].value_counts().to_dict()}")
print(f"  est total length: {url_df['length_sec'].sum()/3600:.1f}h")

url_df.to_csv(OUT_DIR / "_url_filtered.csv", index=False)
"""),

        code_cell("download", """# ============================================================
# Cell 5: Polite audio download (resume + safety limits)
# ============================================================
def safe_species_dir(name):
    return str(name).replace(" ", "_").replace("/", "_").replace("'", "")

url_df["species_dir"] = url_df["scientific_name"].apply(safe_species_dir)
url_df["save_path"] = url_df.apply(
    lambda r: AUDIO_DIR / r["species_dir"] / f"XC{r['xc_id']}.mp3",
    axis=1,
)

# Resume detection
url_df["already_dl"] = url_df["save_path"].apply(
    lambda p: p.exists() and p.stat().st_size > 1000
)
n_existing = int(url_df["already_dl"].sum())
n_to_dl = len(url_df) - n_existing
print(f"Already downloaded: {n_existing}/{len(url_df)}")
print(f"To download: {n_to_dl}")

stats = {
    "ok": 0, "skip_existing": n_existing,
    "fail_total": 0, "fail_4xx": 0, "fail_5xx": 0,
    "fail_timeout": 0, "fail_other": 0,
    "bytes_dl_session": 0,
    "fail_urls": [],
}
start_t = time.time()
to_dl = url_df[~url_df["already_dl"]].reset_index(drop=True)
print(f"Estimated DL time: {len(to_dl) * DL_DELAY_SEC / 3600:.1f}h (at {DL_DELAY_SEC}s/req)")

SAVE_EVERY = 200

def save_progress_state():
    state_df = url_df[["xc_id", "scientific_name", "file_url", "save_path", "already_dl"]].copy()
    state_df["save_path"] = state_df["save_path"].astype(str)
    state_df["downloaded_now"] = state_df["save_path"].apply(
        lambda p: Path(p).exists() and Path(p).stat().st_size > 1000
    )
    state_df.to_csv(OUT_DIR / "_dl_state.csv", index=False)

try:
    for idx, row in tqdm(to_dl.iterrows(), total=len(to_dl), desc="DL"):
        # Safety: runtime
        elapsed_h = (time.time() - start_t) / 3600
        if elapsed_h > MAX_RUNTIME_HOURS:
            print(f"\\n⚠ Hit runtime cap {MAX_RUNTIME_HOURS}h, stopping. Resume next session.")
            break

        # Safety: storage (poll every 50 files)
        if (idx + 1) % 50 == 0:
            try:
                total_gb = sum(f.stat().st_size for f in AUDIO_DIR.rglob("*.mp3")) / 1e9
                if total_gb > STORAGE_CAP_GB:
                    print(f"\\n⚠ Storage cap reached ({total_gb:.2f} GB > {STORAGE_CAP_GB}). Stopping.")
                    break
            except Exception:
                pass

        save_path = Path(row["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        url = row["file_url"]

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
                        last_err = f"too small ({sz}B)"
                        raise ValueError(last_err)
                    stats["ok"] += 1
                    stats["bytes_dl_session"] += sz
                    success = True
                    break
                elif 400 <= r.status_code < 500:
                    last_err = f"HTTP {r.status_code}"
                    stats["fail_4xx"] += 1
                    break  # 4xx は retry 無効
                elif 500 <= r.status_code < 600:
                    last_err = f"HTTP {r.status_code}"
                    stats["fail_5xx"] += 1
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

        time.sleep(DL_DELAY_SEC)

        if (idx + 1) % SAVE_EVERY == 0:
            elapsed = time.time() - start_t
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remain_s = (len(to_dl) - idx - 1) / rate if rate > 0 else 0
            total_gb = stats["bytes_dl_session"] / 1e9
            print(f"  [{idx+1}/{len(to_dl)}] ok={stats['ok']} fail={stats['fail_total']} "
                  f"sess_GB={total_gb:.2f} rate={rate*60:.1f}/min ETA={remain_s/3600:.1f}h")
            save_progress_state()

except KeyboardInterrupt:
    print("\\n⚠ Interrupted")
except Exception as e:
    print(f"\\n⚠ Error: {e}")
    import traceback; traceback.print_exc()

save_progress_state()
print(f"\\n{'='*60}\\nDL session summary\\n{'='*60}")
for k, v in stats.items():
    if k != "fail_urls":
        print(f"  {k}: {v}")
print(f"  Session time: {(time.time()-start_t)/60:.1f} min")
"""),

        code_cell("verify", """# ============================================================
# Cell 6: Verify + final metadata
# ============================================================
url_df["downloaded"] = url_df["save_path"].apply(
    lambda p: Path(p).exists() and Path(p).stat().st_size > 1000
)
url_df["file_size_mb"] = url_df["save_path"].apply(
    lambda p: Path(p).stat().st_size / 1e6 if Path(p).exists() else 0.0
)

n_dl = int(url_df["downloaded"].sum())
total_gb = url_df["file_size_mb"].sum() / 1024
print(f"=== Final state ===")
print(f"  Total URLs: {len(url_df)}")
print(f"  Downloaded: {n_dl} ({100*n_dl/len(url_df):.1f}%)")
print(f"  Total size: {total_gb:.2f} GB")

sp_summary = url_df.groupby("scientific_name").agg(
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

fully = (sp_summary["completion_pct"] == 100).sum()
partial = ((sp_summary["completion_pct"] < 100) & (sp_summary["completion_pct"] > 0)).sum()
none = (sp_summary["completion_pct"] == 0).sum()
print(f"\\n  Fully downloaded: {fully}/{len(sp_summary)} species")
print(f"  Partial: {partial}")
print(f"  Not started: {none}")

# Save final metadata
final_meta = url_df[url_df["downloaded"]][[
    "xc_id", "scientific_name", "genus", "species", "en_name", "type",
    "quality", "length_sec", "country", "license", "save_path", "file_size_mb"
]].copy()
final_meta["filename"] = final_meta["save_path"].apply(
    lambda p: str(Path(p).relative_to(AUDIO_DIR))
)
final_meta = final_meta.drop(columns=["save_path"])
final_meta.to_csv(OUT_DIR / "_metadata.csv", index=False)
sp_summary.to_csv(OUT_DIR / "_species_summary.csv", index=False)

print(f"\\nSaved: {OUT_DIR / '_metadata.csv'}")
print(f"Saved: {OUT_DIR / '_species_summary.csv'}")
"""),

        code_cell("save_dataset", """# ============================================================
# Cell 7: (Optional) Save as Kaggle Dataset
# ============================================================
import json, shutil
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "birdclef2026-xc-api-aves-audio"
TITLE = "BirdCLEF2026 XC API Aves Audio"
USER = "maekeso"

DRY_RUN = True  # ★ confirm before upload

if not DRY_RUN:
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        api.dataset_create_version(folder=str(OUT_DIR), version_notes="initial XC API DL",
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
    print(f"DRY_RUN=True, skip upload")
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
