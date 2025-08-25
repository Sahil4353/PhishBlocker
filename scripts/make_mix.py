import pandas as pd

# load sources
enron = pd.read_csv(r"data/processed/enron_parsed.csv", usecols=["body_text","label"])
spam  = pd.read_csv(r"data/processed/spamassassin_parsed.csv", usecols=["body_text","label"])
phish = pd.read_csv(r"data/processed/nazario_phishing.csv", usecols=["body_text","label"])

# normalize labels (ham/safe -> safe)
canon = {"ham":"safe","ok":"safe","legit":"safe","safe":"safe","spam":"spam","phish":"phishing","phishing":"phishing","fraud":"phishing"}
enron["label"] = enron["label"].astype(str).str.lower().map(lambda x: canon.get(x, x))

# downsample safe (tune N as you like)
safe_sample = enron[enron["label"]=="safe"].sample(150_000, random_state=42)

out = pd.concat([safe_sample, spam, phish], ignore_index=True)
out = out.sample(frac=1.0, random_state=42)  # shuffle
out.to_csv(r"data/processed/mix_balanced.csv", index=False)
print(out["label"].value_counts())
