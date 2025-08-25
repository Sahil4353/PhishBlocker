import os
from pathlib import Path
from subprocess import PIPE, run

import joblib
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent
MODELS = ROOT / "models"


def test_end_to_end_train(tmp_path):
    # tiny synthetic dataset
    df = pd.DataFrame(
        {
            "body_text": [
                "reset your password now",  # phishing-ish
                "verify your account urgently",  # phishing-ish
                "win a free gift voucher!!!",  # spam
                "cheap pills buy now $$$",  # spam
                "meeting agenda attached",  # safe
                "invoice for your order attached",  # safe
            ],
            "label": ["phishing", "phishing", "spam", "spam", "safe", "safe"],
        }
    )
    d1 = tmp_path / "toy.csv"
    df.to_csv(d1, index=False)

    out = tmp_path / "toy_model.joblib"
    cmd = [
        "python",
        str(ROOT / "scripts" / "train_baseline.py"),
        "--inputs",
        str(d1),
        "--val-size",
        "0.34",
        "--seed",
        "123",
        "--max-features-word",
        "5000",
        "--max-features-char",
        "5000",
        "--out",
        str(out),
    ]
    r = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    assert r.returncode == 0, r.stderr

    assert out.exists()
    bundle = joblib.load(out)
    assert "pipeline" in bundle and "label_encoder" in bundle and "metadata" in bundle


def test_predict_script(tmp_path):
    # reuse artifact from previous test if present, else make a quick one
    model = tmp_path / "quick.joblib"
    if not model.exists():
        df = pd.DataFrame(
            {
                "body_text": ["verify account", "hello team meeting", "free prize now"],
                "label": ["phishing", "safe", "spam"],
            }
        )
        csv = tmp_path / "quick.csv"
        df.to_csv(csv, index=False)
        run(
            [
                "python",
                str((ROOT / "scripts" / "train_baseline.py")),
                "--inputs",
                str(csv),
                "--out",
                str(model),
            ],
            check=True,
        )

    # smoke predict
    r = run(
        [
            "python",
            str(ROOT / "scripts" / "predict.py"),
            "--model",
            str(model),
            "please verify your account to continue",
        ],
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "Predicted:" in r.stdout
