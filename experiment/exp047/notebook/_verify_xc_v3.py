"""Verify the v3 API NB has the key + correct URL."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl.ipynb", encoding="utf-8"))

CHECKS = [
    ("setup", "XC_API_KEY = "),
    ("setup", "api/3/recordings"),
    ("query_xc", "XC_API_URL"),
    ("query_xc", "\"key\": XC_API_KEY"),
    ("aves_list", "__BC2026_AVES_RAW = ["),
    ("aves_list", "'Hylophilus pectoralis'"),  # first species
    ("aves_list", "'Tolmomyias sulphurescens'"),  # last species
]

NEG_CHECKS = [
    ("query_xc", "api/2/recordings"),  # v2 should NOT be referenced anymore
]

for cell_id, expected in CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    found = expected in src
    print(f"  {'✓' if found else '✗'} [{cell_id}] {expected[:60]}")

print("\n--- NEG (should NOT exist) ---")
for cell_id, bad in NEG_CHECKS:
    src = ""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            src = "".join(c["source"])
            break
    has_bad = bad in src
    print(f"  {'✗ LEFTOVER' if has_bad else '✓ removed'}: [{cell_id}] {bad}")
