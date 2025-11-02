This folder holds cleaned, normalized data derived from data/raw.

All processed files MUST follow this schema:
- body_text : str (the email/plaintext body)
- label     : one of ["safe","spam","phishing"]

Intermediate files:
- clean_enron.parquet          (mostly "safe")
- clean_spamassassin.parquet   ("spam")
- clean_phishing.parquet       ("phishing")
Final merged training set:
- mix_balanced.parquet         (balanced ~100k per class, deduped)
