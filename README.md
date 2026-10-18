# Text Mining on Mobile Phone Reviews

Text Mining assignment comparing **Samsung** and **Xiaomi** mobile phone listings scraped from
Flipkart. The pipeline cleans and merges the two raw datasets, builds a text document per
phone, preprocesses it, and extracts **Bag-of-Words** and **TF-IDF** features with
exploratory charts.

## Contents

| File | Description |
|---|---|
| `tm assignment 2.py` | Main script — full pipeline (cleaning → preprocessing → features → plots) |
| `samsung_mobiles.csv` | Raw Samsung data (name, ratings, price, camera, battery, storage, ram, processor, android_version) |
| `xiaomi_mobiles.csv` | Raw Xiaomi data (model_names, ratings, storage_ram, os_processor, network, battery) |
| `cleaned_mobile_dataset.csv` | Merged + cleaned dataset with `text`, `clean_text`, `doc_length` columns |
| `bow_features.csv` | Bag-of-Words feature matrix (top 100 unigrams + bigrams) |
| `tfidf_features.csv` | TF-IDF feature matrix (top 100 unigrams + bigrams) |
| `figs/` | Generated charts |

### Charts in `figs/`

- `figs_doc_length.png` — document length distribution
- `figs_brand_counts.png` — record count per brand
- `figs_ratings_distribution.png` — ratings histogram
- `figs_bow_top_tokens.png` — top 15 BoW tokens by frequency
- `figs_tfidf_top_tokens.png` — top 15 TF-IDF features by mean score

## Pipeline

1. **Load & clean** — read both CSVs, tag a `brand` column, drop duplicates, strip whitespace.
2. **Schema harmonisation** — `remap()` maps inconsistent column names (`name`/`model_names`,
   `ram`/`storage_ram`, `processor`/`os_processor`, …) onto a common schema.
3. **Document building** — `compose_text()` joins brand, model, processor, camera, battery and
   RAM into one text document per phone.
4. **Preprocessing** — lowercasing, non-alphanumeric removal, stopword removal, WordNet
   lemmatization.
5. **EDA** — document length stats, missing values, brand counts, ratings distribution.
6. **Feature extraction** — `CountVectorizer` and `TfidfVectorizer` with
   `max_features=100`, `ngram_range=(1,2)`.
7. **Export** — writes the cleaned dataset, both feature matrices and PNG charts.

## Requirements

```
pandas
matplotlib
nltk
scikit-learn
```

## Run

```bash
python "tm assignment 2.py"
```

NLTK downloads `stopwords` and `wordnet` automatically on first run.
Outputs are written to the project folder; charts go to `figs_*.png` (move them into `figs/`
if you want to keep the folder tidy).
