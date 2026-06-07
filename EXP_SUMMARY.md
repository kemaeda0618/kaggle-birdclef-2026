# BirdCLEF+ 2026 実験サマリー

最終更新: 2026-06-03(締切前日)

## 現状

**🥈 LB 0.952(exp110)= 確定天井。堅実 silver(rank ~91 / 4167)**

- 構成: **NB4(LightProtoSSM/Perch)+ Tucker SED 5-fold + exp029 R3(eca_nfnet_l1)** の 3-way rank blend
- weight: **0.30 / 0.40 / 0.30**
- weight / prior / FCS / amphibian の全 PP・blend 軸を探索し尽くし、**0.952 が天井と確定**

締切: **2026-06-04 08:59 JST**(= 2026-06-03 23:59 UTC)

### 最終提出 2 枠(decorrelated hedge)

| 枠 | NB | 構成 | LB |
|---|---|---|---|
| 1 | **exp110** | fold0 e29、weight 30/40/30 | **0.952** |
| 2 | **exp146** | **[1,2] e106 e29**、weight 33/33/33 | **0.952** |

→ e29 stream も weight も異なる = private shake への hedge。**exp148/149(0.948)・exp130(0.925)・exp141(0.942)は絶対選ばない**

## メダル境界(2026-06-02、4167 teams 実測)

| メダル | border LB | 順位 |
|---|---|---|
| 🥇 Gold | 0.958 | top 18 |
| 🥈 Silver | 0.951 | top 208 |
| 🥉 Bronze | 0.950 | top 416 |
| **我々 0.952** | — | **rank ~91**(silver線まで117位、bronze線まで325位のクッション)|

→ silver/bronze border は 05-31 から 2 日間フラット(creep 低)。0.952 は堅実 silver。

---

## 実験記録: exp090〜149 era(2026-05-30 〜 06-03)

### Blend weight sweep(3-way、fold0 e29 base)

| Exp | weight (NB4/Tucker/e29) | LB | 結論 |
|---|---|---|---|
| exp021 | 30 / 45 / 25 | 0.946 | Tucker↑ drag |
| exp019 | 35 / 40 / 25 | 0.947 | NB4↑/e29↓ drag |
| exp139 | 25 / 40 / 35 | 0.950 | e29↑ drag |
| exp140 | 27.5 / 40 / 32.5 | 0.950 | e29↑ drag |
| exp143 | 33 / 33 / 33(均等)| 0.951 | Tucker↓ で -0.001 |
| **exp110** | **30 / 40 / 30** | **0.952** | **★ peak** |

→ **30/40/30 が全方向 peak。weight 軸 飽和確定。** 表面は flat plateau(0.951-0.952、public noise floor)。

### e106 multi-fold swap(e29 stream を exp029 単fold → exp106 5-fold アンサンブルに)

| Exp | e106 fold | Tucker | weight | LB | 備考 |
|---|---|---|---|---|---|
| exp111 | exp102 [1,2] | 5-fold OV | 30/40/30 | 0.946 | INT8 量子化が -0.018 と特定 |
| exp116 | [0,1] | 3-fold OV | 30/40/30 | 0.949 | FP32、Tucker 削減 |
| exp117 | [0,2] | 3-fold OV | 30/40/30 | 0.950 | FP32 |
| exp112 V4 | [1,2] | 5-fold OV | 30/40/30 | 0.951 | fold ペア [1,2] が最良 |
| exp144 | [0,1] | 5-fold OV | 30/40/30 | (COMPLETE, 未sub) | 最悪ペア |
| exp145 | [1,2] | 5-fold OV | 27.5/40/32.5 | 0.951 | e29 boost 効かず |
| **exp146** | **[1,2]** | **5-fold OV** | **33/33/33** | **0.952** | **★ exp110 と同点、decorrelated 2nd pick** |
| exp147 | [1,2] | 5-fold OV | 50/0/50(Tucker抜き)| (COMPLETE, 未sub) | 診断用 |

→ 学び: **fold ペアは noise でなく差がある**([1,2]=最良 > [0,2] > [0,1])。**最適 weight は e29 stream 依存**(fold0→30/40/30、[1,2]→33/33/33 が最適)。e106 path は 0.952 で頭打ち。**3-fold + Tucker5 は timeout(~100min)**。

### Amphibian specialist surgical blend(蛙列のみ差し替え)

| Exp | specialist | gate(held-out col-AUC)| α | LB | 結論 |
|---|---|---|---|---|---|
| exp130 | v1(weak label)| 0.655 | 0.5 | 0.925 | FP 大量、大失敗 |
| exp141 | v5(Pantanal+AnuraSet)| 0.798 | 0.5 | 0.942 | v1 より良いが base に負け |

→ **amphibian blend 軸 死亡。** base のアンサンブルが、我々が作れるどの蛙 specialist よりも実 test で強い(base が labeled-SC で訓練済 → specialist と冗長)。held-out gate(同サイト漏れ)は LB を予測しない。詳細は memory `project_amphibia_anuraset_finding`。

### Post-process: post-blend site/hour prior(exp110 に prior を blend 後に追加)

| Exp | prior 位置 | λ | LB | 結論 |
|---|---|---|---|---|
| exp148 | 最初(rank-blend 直後)| 0.30 | 0.948 | 順序 suboptimal と当初分析 |
| exp149 | 最後(mirror→FCS→tax の後)| 0.30 | 0.948 | **順序を直しても同じ** |

→ **post-blend prior 軸 死亡。** 両方 0.948(-0.004)= 順序は無関係、**prior 自体が drag**。原因: NB4 stream に既に site/hour prior(λ=0.65)が入っており冗長 + 全 Pantanal で site/hour 変動が弱く、λ=0.30 で過大評価。数学分析(順序が原因)は実データに否定された。

---

## 実験記録: exp010〜088 era(2026-05-09 〜 28、要約)

### Single-stream standalone LB

| Exp | Backbone / 概念 | LB | 採否 |
|---|---|---|---|
| exp010 / NB4 v11 | Perch + ProtoSSM + MLP | 0.926 | ✅ blend slot 1 |
| exp037 v313 | LightProtoSSM + MLP + ResSSM | 0.930 | ✅ blend slot 1(現行)|
| exp017 | eca_nfnet_l0 R2 | 0.921 | ✅(旧 slot 3)|
| exp029 | eca_nfnet_l1 R3 single | 0.923 | ✅ blend slot 3(現行)|
| exp014 | 自前 SED(HGNet+Perch distill)| 0.907 | ❌ pseudo memorize |
| exp015 | convnext_pico R2 | 0.910 | ❌ ensemble drag |
| exp016 | regnety_008 R2 | 0.903 | ❌ |
| exp032 | effnet_b3 + Babych mel | 0.887/0.786 | ❌ pretrain mismatch |
| exp036 | Soft AUC fine-tune | 0.914 | ❌ val/test divergence |
| exp076 (M7) | Babych b0 + 63 sub | 0.904 | ✅ Babych transfer 実証 |
| exp013 | hgnet_r1 standalone | 0.729 | ❌ low diversity |

### Blend / PP era

| Exp | 構成 | LB | 結論 |
|---|---|---|---|
| exp019 | NB4+Tucker+e17 R2(35/40/25)| 0.947 | silver 初到達 |
| exp031 | NB4+Tucker+exp029 R3(35/40/25)| 0.949 | e17→exp029 swap |
| exp048 | exp040 + TopN N=1 PP | 0.950 | FCS spec(K=1/POW=1.0)= +0.001、FCS 飽和実証 |
| exp067 | + Pantanal proximity | drag | geo prior は site prior と冗長 |
| exp068 | + Delta TTA | 0.950 (±0) | post-process 飽和 |
| exp078 | exp037+Tucker+exp029(35/40/25)| 0.950 | NB4→exp037 swap |
| exp085 | exp078 + exp015 ConvNeXt | 0.949 | 同 mel/paradigm drag |
| exp023 | species wet/dry prior | 0.929 (-0.018) | sparse すぎ大失敗 |
| exp026 | NFNet 5-fold weight-avg | 0.929 (-0.018) | weight avg 破壊 |
| ensemble e14-17 | 4 Tucker backbone equal | 0.931 | +0.018 ensemble bonus 実証 |

---

## 軸別 最終結論(2026-06-03)

| 軸 | 状態 | 根拠 |
|---|---|---|
| weight | **飽和**(30/40/30 peak)| exp019/021/139/140/143 全て ≤0.952 |
| post-blend prior | **死亡** | exp148/149 両方 0.948 |
| FCS | **ほぼ飽和** | exp048 で +0.001 のみ |
| amphibian blend | **死亡** | exp130=0.925、exp141=0.942 |
| e106 multi-fold | **0.952 頭打ち** | exp146=0.952、3-fold は timeout |
| ConvNeXt/RegNet 追加 | **drag** | exp085 -0.001、diversity 飽和 |
| geo/wet-dry prior | **drag** | exp067/023 |

## 主要な学び

- **NFNet 系は test 域 generalization 圧倒的**(exp017 gap -0.004)→ blend で重視
- **rank blend > logit blend**(AUC 整合)
- **ensemble bonus は backbone family diversity > variant > mel > window**
- **同 mel + 同 paradigm の追加は drag**(exp085)
- **弱い single(LB<0.85)の blend 加算は drag**(exp055/56/57)
- **fold ペアは noise でなく差がある**([1,2]>[0,2]>[0,1])、**最適 weight は stream 依存**
- **後付け blend は data-rich な base に勝てない**(amphibian で実証)、外部データは訓練に統合すべき(= gold gap)
- **BCE が optimal**、ASL/Focal/CE/Soft AUC は機能せず
- **PP の適用順序が効く場合がある**(prior は metric が rank-based なので、global 頻度は列内一律=無影響、site/hour 変動のみ効く)
- **検証前に「未テスト」と断言しない**(exp106 fold ペア、exp148 順序、per-taxon=exp056 で繰り返しミス)→ 組む前に設計分析

## 残る唯一の未探索レバー(低-中確率)

| 候補 | 内容 | 評価 |
|---|---|---|
| per-taxon blend(reweight 版)| 既存3 stream の重みをクラス別に(非 Aves に Tucker 厚め)。公開 0.948 NB は mapped/unmapped per-class blend を使用 | 低-中。但し exp056 が per-taxon overlay(Aves stage2a)を試して abandoned(弱い negative)|
| FCS K=1/POW=1.0 を exp110 に | exp048 推奨だが exp110 未適用 | 低(+0.001、飽和)|
| clip/frame 融合比(0.5/0.5)| Tucker/e29 の出力合成比 | 低、未 sweep |

→ いずれも ±0.001-0.003、plateau 内。**0.953 保証なし。0.952 が天井である可能性が高い。**

## ❌ 非推奨パス(撤退確定)

- BirdNET / AVES / 新 paradigm 軸(user 明示)
- 同 stream tuning だけで gold 突破(0.952 plateau)
- iNat/AnuraSet pretrain 単純使用(catastrophic forget)
- Soft AUC / Asymmetric / Focal / CE loss
- architecture single upgrade(effect 0)
- wet/dry / day-of-year / geo-proximity metadata 軸
- 弱い single の blend 加算
- 同 mel + 同 backbone family の追加
- **post-blend site/hour prior(λ無関係に -0.004)**
- **amphibian surgical blend(base が specialist より強い)**

## ファイル参照

- 現 best NB: `experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb`
- 2nd pick NB: `experiment/exp146/notebook/nb_exp146_fold12_w333.ipynb`
- memory index: `~/.claude/.../memory/MEMORY.md`(特に `project_ceiling_saturation_20260603`)
