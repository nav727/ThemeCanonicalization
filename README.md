# Theme Discovery and Canonicalisation in Financial News

Three keyphrases, one common theme, but shows up as three separate nodes in your knowledge graph.
- "Fed rate hike"
- "Federal Reserve raises interest rates" and
- "Tightening by the Fed"

<img width="739" height="336" alt="Knowledge graph: before and after theme canonicalization" src="https://github.com/user-attachments/assets/6c51d39d-fce9-4fb4-9b1f-ae4ecba115a9" />

A human reads those and sees one story. A knowledge graph built from these news articles sees three unrelated nodes and every task downstream (for example ['thematic investing'](https://www.blackrock.com/us/individual/insights/thematic-investing) leveraging this KG) suffers because of this.

Take for example, in late 2021, the same supply chain crunch was written up as "port congestion" by one news outlet, "logjams" by another and "supply snarls" by a third. The article count splits three ways, so no single node ever crossed an 'alerting threshold'. A real disruption went unnoticed (at least to the system). Merge the three onto one node and the spike is obvious.

This repo is my MSc thesis that tries to fix for this problem. An LLM reads each article and returns the keyphrases in it plus a knowledge graph built around them. Those keyphrases are then clustered onto one canonical theme node.

Research Questions:-
-  Whether the keyphrase text + modern embedding model is enough for clustering?
-  Test whether the graph structure around the keyphrase can improve on canonicalization task.

<img width="1458" height="721" alt="sol_overview" src="https://github.com/user-attachments/assets/5e3ac7f9-1fda-4e88-8e74-f3bf35af27e2" />


<br />

🟥 Spoiler: They were not! Keyphrase text alone scores 0.438 F1 on the held out test split; adding knowledge graph entities takes it to 0.870.

## Pipeline

Run the notebooks in numeric order; each one consumes the previous one's output in `data/`.

| Notebook | Purpose |
| --- | --- |
| [1_preprocess_dataset.ipynb](notebooks/1_preprocess_dataset.ipynb) | Merge the per-year JSON dumps, drop price columns, add `article_id` → `combined_2017_2023_cleaned.csv` |
| [2_extract_themes_KG_LLM.ipynb](notebooks/2_extract_themes_KG_LLM.ipynb) | DeepSeek theme + KG extraction (async, checkpointed, resumable), then ontology cleaning of the raw graphs |
| [3_synthetic_golden_set.ipynb](notebooks/3_synthetic_golden_set.ipynb) | LLM-generated paraphrase variants of real KGs → `golden_pairs.csv`, `synthetic_gold_themes.csv` |
| [4_fine_tune_llm.ipynb](notebooks/4_fine_tune_llm.ipynb) | QLoRA distil of the teacher labels into Qwen2.5-7B-Instruct, with before/after eval on a held-out test set |
| [5_Exp1 — Chap 4](notebooks/5_Exp1_textual_similarity_clustering_baseline_Chap4.ipynb) | Baseline: Jaccard word overlap blended with bge-m3 phrase embeddings, HAC / Leiden clustering |
| [6_Exp2 — Chap 5](notebooks/6_Exp2_entity_weighted_theme_representations_Chap5.ipynb) | Entity-weighted theme vectors, `w(e,θ) = BM25 · IDF · type_prior`, optional 1-hop KG propagation |
| [7_Exp3 RGCN](notebooks/7_Exp3_Graph_Neural_Networks_Theme_Representations_Chap6_RGCN.ipynb) · [8_Exp3 RGAT](notebooks/8_Exp3_Graph_Neural_Networks_Theme_Representations_Chap6_RGAT.ipynb) — Chap 6 | Relational GNN theme embeddings + two-stage clustering; RGAT adds attention-entropy, relation-ablation and edge-mask interpretability |



## Corpus & Evaluation

Corpus contains 61,752 English financial news articles (2017 – 2023). 5,433 of them (8.8%) went through LLM extraction, yielding 17,201 keyphrases and 87,969 KG triplets.

All three experiments are scored the same way, so the chapters are directly comparable: pairwise
precision / recall / F1 and ARI over labelled theme pairs in `data/synthetic_gold_themes.csv`,
using the cluster-disjoint split in `data/synthetic_gold_split.csv` (written by notebook 7 with
`SEED = 72`, reused by 5, 6 and 8). Configurations are selected on `val`; `test` is untouched
until the final number. Each experiment writes a `canon_*.csv` of theme → cluster assignments and
an identical three-CSV report under `output/<experiment>/report`, so any cluster can be read end to
end: its label, its member themes, the KG triples behind them, and the source articles.

Dataset and evaluation files can be found --> [`MannSingh/financial-news-theme-kg`](https://huggingface.co/datasets/MannSingh/financial-news-theme-kg)


## Layout

```
notebooks/           the pipeline above (run from this directory; paths resolve as ../data)
data/                inputs and intermediates (gitignored)
output/              per-experiment results: eda, exp1_chapter4, exp2_chapter5,
                     exp3_chapter6_rgcn, exp3_chapter6_gat
llm_finetune_output/ QLoRA adapter (kg-lora) and the train/val/test JSONL splits
```

Note: Data, outputs, figures and model weights are all gitignored



## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the repo root (never commit it) with:

| Key | Used by |
| --- | --- |
| `DEEPSEEK_API_KEY` | notebook 2 (theme + KG extraction) |
| `OPENROUTER_API_KEY` | notebook 3 (synthetic golden set) |

Two things to adjust before running locally: notebook 2 has a hardcoded absolute `ENV_PATH`, and
notebooks 4 and 11 have an `env = "GPU"` toggle at the top pointing at cluster paths.

## Data and licence

The base articles come from
[FelixDrinkall/financial-news-dataset](https://github.com/FelixDrinkall/financial-news-dataset),
licensed **CC BY-NC-SA 4.0**. ShareAlike means this derivative carries the same licence and must
keep the attribution — a licence condition, not a courtesy.

## Artefacts

| Artefact | Link |
| --- | --- |
| Dataset and evaluation dataset | [`MannSingh/financial-news-theme-kg`](https://huggingface.co/datasets/MannSingh/financial-news-theme-kg) |
| Fine-tuned LLM model | [`MannSingh/qwen2.5-7b-theme-kg-lora`](https://huggingface.co/MannSingh/qwen2.5-7b-theme-kg-lora) |

The model repo is adapter-only — load `Qwen/Qwen2.5-7B-Instruct` and apply the LoRA on top with
`PeftModel`.
