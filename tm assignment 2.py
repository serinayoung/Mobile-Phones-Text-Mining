import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

# datasets used
SAMSUNG_FILE = "samsung_mobiles.csv"
MI_FILE = "xiaomi_mobiles.csv"

samsung = pd.read_csv(SAMSUNG_FILE)
xiaomi  = pd.read_csv(MI_FILE)

samsung["brand"] = "Samsung"
xiaomi["brand"]  = "Xiaomi"

# cleaning function
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for col in df.columns:
        if df[col].dtype == object:
            # Use .str.strip() to handle entire Series
            df[col] = df[col].astype(str).str.strip()
    return df

samsung = basic_clean(samsung)
xiaomi  = basic_clean(xiaomi)

ALIASES = {
    'model':    ['Model','model','name','model_names'],
    'ram':      ['RAM','ram','storage_ram'],
    'battery':  ['Battery','battery'],
    'camera':   ['Camera','camera'],
    'processor':['Processor','processor','os_processor']
}

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def remap(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for target, cands in ALIASES.items():
        src = find_col(df, cands)
        out[target] = df[src] if src else ""
    out['brand'] = df['brand'] if 'brand' in df.columns else ""
    # optional fields
    ratings_col = find_col(df, ['ratings','Ratings','rating'])
    out['ratings'] = df[ratings_col] if ratings_col else pd.NA
    return out

s_df = remap(samsung)
m_df = remap(xiaomi)

data = pd.concat([s_df, m_df], ignore_index=True)
print("Total records:", len(data))

# text document
text_columns = ["model", "ram", "battery", "camera", "processor"]
data[text_columns] = data[text_columns].astype(str)

def compose_text(row):
    parts = []
    if row['brand']: parts.append(row['brand'])
    if row['model']: parts.append(row['model'])
    if row['processor']: parts.append(f"runs {row['processor']}")
    if row['camera']: parts.append(f"camera {row['camera']}")
    if row['battery']: parts.append(f"battery {row['battery']}")
    if row['ram']: parts.append(f"ram {row['ram']}")
    return ". ".join(parts)

data["text"] = data.apply(compose_text, axis=1)

# text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    t = str(text).lower()
    t = re.sub(r'[^a-z0-9]+', ' ', t)  
    toks = t.split()
    toks = [lemmatizer.lemmatize(w) for w in toks if w not in stop_words]
    return " ".join(toks)

data["clean_text"] = data["text"].apply(preprocess)

Path("figs").mkdir(exist_ok=True)

data["doc_length"] = data["clean_text"].apply(lambda x: len(x.split()))
print("\nDocument length stats:")
print(data["doc_length"].describe())

plt.figure(figsize=(7,4))
data["doc_length"].hist(bins=30, color="#1976d2")
plt.title("Document Length Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figs_doc_length.png")
plt.close()

print("\nMissing values per column:")
print(data.isna().sum())

plt.figure(figsize=(6,4))
data["brand"].value_counts().plot(kind="bar", color=["#1428a0","#ff6700"])
plt.title("Records by Brand")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figs_brand_counts.png")
plt.close()

if data['ratings'].notna().any():
    plt.figure(figsize=(7,4))
    data['ratings'].dropna().plot(kind='hist', bins=20, color="#2e7d32")
    plt.title("Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figs_ratings_distribution.png")
    plt.close()

# bag of words
bow = CountVectorizer(
    max_features=100,
    ngram_range=(1,2),
    token_pattern=r'(?u)\b\w+\b'  
)
bow_matrix = bow.fit_transform(data["clean_text"])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names_out())

print("\nBag-of-Words — sample top features:")
print(pd.Series(bow.get_feature_names_out()).head(20))

plt.figure(figsize=(8,4))
pd.Series(bow_df.sum()).sort_values(ascending=False).head(15).plot(kind='bar', color="#6a1b9a")
plt.title("Top BoW Tokens")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figs_bow_top_tokens.png")
plt.close()

# tfidf
tfidf = TfidfVectorizer(
    max_features=100,
    ngram_range=(1,2),
    token_pattern=r'(?u)\b\w+\b'
)
tfidf_matrix = tfidf.fit_transform(data["clean_text"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

top_tfidf = tfidf_df.mean().sort_values(ascending=False).head(15)
print("\nTF-IDF — top 15 features (mean score):")
print(top_tfidf)

plt.figure(figsize=(8,4))
top_tfidf.plot(kind="bar", color="#c62828")
plt.title("Top TF-IDF Tokens")
plt.ylabel("Average TF-IDF Score")
plt.tight_layout()
plt.savefig("figs_tfidf_top_tokens.png")
plt.close()

# outputs
bow_df.to_csv("bow_features.csv", index=False)
tfidf_df.to_csv("tfidf_features.csv", index=False)
data.to_csv("cleaned_mobile_dataset.csv", index=False)

print("\nOutputs saved successfully:")
print("- cleaned_mobile_dataset.csv")
print("- bow_features.csv")
print("- tfidf_features.csv")
print("- figs_.png charts")
