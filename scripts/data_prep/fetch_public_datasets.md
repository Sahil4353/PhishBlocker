Datasets to download manually:

1. Enron Email Dataset (ham / safe)
   - Commonly distributed as "Enron email dataset" (e.g. CMU / Legal Aid / Kaggle mirrors)
   - Download the full maildir dump (~400MB+), or a cleaned CSV/Parquet version
   - Place under data/raw/enron/

   Goal label: "safe"

2. SpamAssassin Public Corpus
   - Commonly published as spamassassin "easy_ham", "hard_ham", "spam" mbox files
   - Download spamassassin public corpus (there are mirrors on Apache / Kaggle)
   - Unzip into data/raw/spamassassin/
   - We'll treat:
        spam/*          → "spam"
        easy_ham/*      → "safe"
        hard_ham/*      → "safe"

3. Nazario Phishing Corpus
   - Jose Nazario's phishing corpus (URLs / phishing mails, historically mirrored online)
   - Download and unzip into data/raw/nazario/
   - We'll treat all Nazario samples as "phishing"

After downloading:
- DO NOT commit data/raw.
- We'll run the data_prep scripts to build cleaned parquet files under data/processed.
