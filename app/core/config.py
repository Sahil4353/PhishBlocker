import os
from pathlib import Path


class Settings:
    """
    Lightweight settings with env overrides.
    No extra deps needed. Example env:
      ENV=prod
      ENABLE_LOGGING=false
      LOG_LEVEL=DEBUG

    New:
      MODEL_PATH=models/tfidf_lr_small_l2.joblib
      MODEL_METRICS_PATH=models/tfidf_lr_small_l2.metrics.json
      MODEL_VERSION=tfidf_lr_small_l2@2025-10-27
    """

    ENV: str = os.getenv("ENV", "dev")

    ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- ML model config ---
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        # sensible default for local dev
        str(Path("models") / "tfidf_lr_small_l2.joblib"),
    )

    MODEL_METRICS_PATH: str = os.getenv(
        "MODEL_METRICS_PATH",
        str(Path("models") / "tfidf_lr_small_l2.metrics.json"),
    )

    MODEL_VERSION: str = os.getenv(
        "MODEL_VERSION",
        "",  # if empty we'll fall back to artifact filename stem
    )


settings = Settings()
