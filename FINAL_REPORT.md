# Word2Vec 対義語空間分析 - 最終レポート

## 概要

Word2Vecの単語ベクトル空間において、対義語ペアの関係性を利用した新しい意味表現システムを構築しました。対義語方向を「意味の座標系」として使用することで、様々な意味操作が可能になります。

## 主要な発見

### 1. 対義語ベクトルは意味の座標系を形成する

```
         +good
           ↑
           |
  +small ←─0─→ +big
           |
           ↓
         +bad
```

- **ゼロ点** = 中立（全対義軸の交点）
- **加算** = 意味成分の追加
- **減算** = 意味成分の除去

### 2. 対義関係は低次元で表現可能

| 累積分散説明率 | 必要次元数 |
|---------------|-----------|
| 50% | 25次元 |
| 70% | 47次元 |
| 90% | **76次元** |
| 99% | 97次元 |

2,352個の対義語ペアが76次元で90%説明可能 → 対義関係には普遍的な構造がある

### 3. 多義語は複数軸で表現可能

**例: "light"の意味軸**

| 軸 | 意味 | 対義語例 |
|----|------|---------|
| dark | 明るさ | dark, dim, shadows |
| heavy | 重さ | heavy, massive, weighty |
| severe | 症状の程度 | chronic, debilitating |
| serious | 態度 | consequences, problems |
| dense | 密度 | thick, sparse, vegetation |
| fat | 栄養 | calories, cholesterol |

### 4. 意味の加算が機能する

#### 強度の加算
| 変換 | 結果 |
|------|------|
| bad + intensity | terrible, horrible ✓ |
| happy + intensity | joyful, ecstatic ✓ |
| cold + intensity | freezing, frigid ✓ |

#### 属性の加算
| 変換 | 結果 |
|------|------|
| car + big | truck, vehicle ✓ |
| news + paper | newspaper ✓ |
| air + port | airport ✓ |

#### 反対方向への変換
| 変換 | 結果 |
|------|------|
| love → hate方向 | hatred, racism ✓ |
| success → failure方向 | failed, failing ✓ |

## 実装した機能

### 基本機能
- **対義語検出**: ベクトル差分法による高精度な対義語検出
- **双方向検証**: A→B かつ B→A の検証
- **高速検索**: 155ms/単語（4.1倍高速化）

### 多義語処理
- **意味ごとの軸分離**: 各意味に対して別々の対義語軸を作成
- **WSD統合**: 文脈から適切な軸を自動選択

### 自動軸発見
- **クラスタリングベース**: WordNet不要で意味軸を発見
- **品質フィルタリング**: ノイズ除去による高精度な軸発見
- **新規発見**: WordNetにない対義関係も発見可能

### 意味演算
- **加算**: 意味成分の追加（word + good方向）
- **減算**: 意味成分の除去（happy - emotion成分）
- **強度変換**: 程度の調整（good → excellent）
- **分解**: 単語を意味成分に分解

## 応用

### 1. 感情分析
```python
sentiment = dot(word_vector, good_bad_axis)
# positive: excellent(+2.57), good(+2.30)
# negative: bad(-1.52), horrible(-1.21)
```

### 2. バイアス検出
```python
gender_bias = dot(occupation_vector, he_she_axis)
# Male-biased: manager(+1.58), engineer(+0.82)
# Female-biased: nurse(-1.60), receptionist(-1.22)
```

### 3. 類義語/対義語の区別
従来のコサイン類似度では区別できなかった問題を解決
- happy-sad: 類似度0.68（高い）だが**対義語**として正しく検出

### 4. 強度グラデーション
```
freezing → cold → cool → lukewarm → warm → hot → boiling
```
順序が自動的に発見される

### 5. 意味的中立性の測定
- 感情語（平均50.3）: 強い意味的極性
- 機能語（平均70.1）: 意味的に中立

## ファイル構成

```
reverse_word2vec/
├── src/
│   ├── antonym_loader.py      # WordNetから対義語を読み込み
│   ├── word2vec_loader.py     # GloVeモデルを読み込み
│   ├── fast_antonym_v2.py     # 高速対義語検出（最終版）
│   └── experiments/
│       ├── polysemy.py        # 多義語処理
│       ├── intensity.py       # 強度分析
│       ├── generation.py      # 対義語生成
│       ├── dimension_analysis.py  # 次元分析
│       ├── model_comparison.py    # モデル比較
│       ├── wsd_antonym.py     # WSD統合
│       ├── auto_axis_discovery.py # 自動軸発見
│       ├── applications.py    # 応用（感情分析等）
│       └── semantic_arithmetic.py # 意味演算
├── notebooks/
│   └── experiments_summary.ipynb  # 実験まとめノートブック
├── EXPERIMENT_REPORT.md       # 詳細な実験レポート
└── FINAL_REPORT.md           # このファイル
```

## 実行方法

```bash
# 環境構築
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 基本的な対義語検出
PYTHONPATH=src python src/fast_antonym_v2.py

# 各種実験
PYTHONPATH=src python src/experiments/polysemy.py
PYTHONPATH=src python src/experiments/auto_axis_discovery.py
PYTHONPATH=src python src/experiments/applications.py

# Jupyterノートブック
jupyter notebook notebooks/experiments_summary.ipynb
```

## 結論

対義語ベクトルは単なる「反対語の発見」以上の可能性を持ちます：

1. **意味の座標系**: 対義語軸は意味空間の基底ベクトルとして機能
2. **意味演算**: 加算・減算による意味の操作が可能
3. **普遍構造**: 対義関係には言語を超えた共通構造がある
4. **実用応用**: 感情分析、バイアス検出、意味分解に応用可能

## 今後の発展

1. **BERTとの統合**: 文脈依存の対義語処理
2. **多言語対応**: 日本語や他言語への適用
3. **デバイアシング**: バイアス軸を使った埋め込み修正
4. **テキスト生成制御**: LLMの出力を意味軸で制御

## 参考文献

1. Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases"
2. Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation"
3. Miller, G. A. (1995). "WordNet: A Lexical Database for English"

---

**Repository**: https://github.com/kasi-x/reverse_word2vec
