# Indonesian Tweet Emotion Classification with Word2Vec

A text processing coursework project for classifying Indonesian tweets into five emotions: **anger, fear, happy, love, and sadness**. The original notebook explored pretrained Word2Vec embeddings with a Random Forest classifier, plus combined TF-IDF and Word2Vec features with Multinomial Naive Bayes.

## What the original notebook recorded

| Item | Recorded value |
|---|---:|
| Dataset rows | 4,401 |
| Train / validation split | 3,520 / 881 (80/20) |
| Word2Vec vector size | 400 |
| Word2Vec + Random Forest validation accuracy | Approximately 35% |
| TF-IDF + Word2Vec + Naive Bayes validation accuracy | Approximately 36% |

These are **historical outputs**, not results produced by this cleaned notebook. The original Word2Vec section iterated over characters in a string instead of word tokens; the rewritten notebook fixes that bug. Its results will differ and depend on the exact local files. The original experiment used a validation split only; it did not establish performance on an independent test set.

## Files needed to run

```text
data/
├── Twitter_Emotion_Dataset.csv  # must contain tweet and label columns
├── Word2Vec_400dim.txt          # pretrained vectors in word2vec text format
└── bahasa_gaul.csv              # optional: slang and formal columns
```

These files are **not included**. The uploaded notebook referred to files on a personal Google Drive but did not record their source or redistribution rights. Use your own authorized copies and check their license before publishing data or embedding weights. The `data/` directory is excluded from Git. If you use Colab, mount Drive and edit the three paths in the configuration cell.

## Run

```bash
python -m venv .venv
# Activate your virtual environment
pip install -r requirements.txt
jupyter notebook indonesian_tweet_emotion_word2vec.ipynb
```

Run cells top to bottom. The notebook loads the local data, cleans and normalizes text, makes average word embeddings, trains Random Forest, compares TF-IDF + embeddings with Naive Bayes, and prints validation classification reports and confusion matrices.

## Notes

- The corrected notebook uses word tokens for Word2Vec lookup and removes broken cells that referenced undefined variables.
- TF-IDF and scaling are fitted on the training portion only.
- Results should be evaluated on a separate held-out test set before making claims about generalization.
