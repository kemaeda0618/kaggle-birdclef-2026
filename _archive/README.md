# _archive — 一回限りの調査スクリプト退避先

BirdCLEF 2026 開発中、リポジトリ直下に溜まった ad-hoc スクリプトと dump を
2026-06-07 にここへ移動（コンペ終了後の整理）。**削除ではなく退避** — 後日の振り返りで
「当時どう調査したか」を辿れるよう残してある。中身は全て一回限りの使い捨てで、
再実行する想定はない（多くは特定 exp 番号・特定 Kaggle kernel に紐づく）。

## 構成
| ディレクトリ | 内容 | 件数 |
|---|---|---|
| `scripts/build/`    | `_build_*.py` — Kaggle NB 生成スクリプト | 8 |
| `scripts/fixes/`    | `_fix_*.py` — NB/アップロードの一回限り修正 | 19 |
| `scripts/analysis/` | `_dump_/_extract_/_search_/_pull_/_summarize_/_analyze_/_species_*.py` — 解析系 | 16 |
| `scripts/checks/`   | `_check_/_inspect_/_verify_/_audit_/_diff_/_compare_/_show_*.py` 等 — 診断系 | 69 |
| `dumps/`            | `_*.txt`, `r1_inspect.txt` — ログ/セルの dump 出力 | 20 |

## 触っていないもの（root にそのまま残した記録）
- `CLAUDE.md` / `EXP_SUMMARY.md` / `BACKLOG.md` / `BACKLOG_PP.md` / `PAST_SOLUTIONS.md`
- `taxonomy.csv` / `sample_submission.csv` / `train.csv` / `train_soundscapes_labels.csv` / `recording_location.txt`
- `experiment/expNNN/`（各実験の NB・config・build script）
- `survey/`（論文・discussion・上位手法メモ）, `_public_nb_pulls/`（上位公開 NB）
