# Theme Discovery and Canonicalisation in Financial News

MSc thesis code. Each news article is passed through an LLM that extracts open-vocabulary
**themes** plus a **knowledge graph** of typed triplets anchored to those themes. The themes are
then canonicalised — free-text phrases such as *"labor union formation effort"* and
*"labor union organizing campaign"* have to collapse into one cluster — using three
representations of increasing structure, each scored against a synthetic golden set.

Corpus: 2017–2023 financial news (~70k articles, ~8.5% sent through LLM extraction).

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
| [9_EDA_raw_news_and_extraction.ipynb](notebooks/9_EDA_raw_news_and_extraction.ipynb) | Corpus and extraction EDA figures |
| [10_push_datasets_to_hf.ipynb](notebooks/10_push_datasets_to_hf.ipynb) · [11_push_model_to_hf.ipynb](notebooks/11_push_model_to_hf.ipynb) | Publish the datasets and the LoRA adapter to Hugging Face (private repos) |

## Evaluation

All three experiments are scored the same way, so the chapters are directly comparable: pairwise
precision / recall / F1 and ARI over labelled theme pairs in `data/synthetic_gold_themes.csv`,
using the cluster-disjoint split in `data/synthetic_gold_split.csv` (written by notebook 7 with
`SEED = 72`, reused by 5, 6 and 8). Configurations are selected on `val`; `test` is untouched
until the final number. Each experiment writes a `canon_*.csv` of theme → cluster assignments and
an identical three-CSV report under `output/<experiment>/report`, so any cluster can be read end to
end: its label, its member themes, the KG triples behind them, and the source articles.

## Layout

```
notebooks/           the pipeline above (run from this directory; paths resolve as ../data)
data/                inputs and intermediates (gitignored)
output/              per-experiment results: eda, exp1_chapter4, exp2_chapter5,
                     exp3_chapter6_rgcn, exp3_chapter6_gat
llm_finetune_output/ QLoRA adapter (kg-lora) and the train/val/test JSONL splits
misc/                earlier exploratory notebooks — not part of the pipeline
Papers/              reference PDFs
```

Data, outputs, figures and model weights are all gitignored, so a fresh clone starts at notebook 1.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins only the core three (`transformers`, `torch`, `datasets`). The notebooks
also use `openai`, `tenacity`, `python-dotenv`, `pandas`, `scikit-learn`, `networkx`,
`sentence-transformers`, `torch-geometric`, `umap-learn`, `peft`, `trl` and `bitsandbytes`.

Create a `.env` in the repo root (never commit it) with:

| Key | Used by |
| --- | --- |
| `DEEPSEEK_API_KEY` | notebook 2 (theme + KG extraction) |
| `OPENROUTER_API_KEY` | notebook 3 (synthetic golden set) |
| `HF_TOKEN` | notebooks 10 and 11 (write access) |

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
