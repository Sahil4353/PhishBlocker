import os


class Settings:
    """
    Lightweight settings with env overrides.
    No extra deps needed. Example env:
      ENV=prod
      ENABLE_LOGGING=false
      LOG_LEVEL=DEBUG
    """

    ENV: str = os.getenv("ENV", "dev")
    ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- ML / Data paths (safe defaults) ---


MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
MODEL_FILE: str = os.getenv("MODEL_FILE", "text_v1.joblib")  # will exist after training
MODEL_PATH: str = os.path.join(MODEL_DIR, MODEL_FILE)

DATA_DIR: str = os.getenv("DATA_DIR", "data")
RAW_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")

ENABLE_ML: bool = os.getenv("ENABLE_ML", "true").lower() == "true"


settings = Settings()
