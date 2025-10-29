import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


class Settings:
    """
    Lightweight settings with env overrides.
    Example .env file:
      ENV=prod
      ENABLE_LOGGING=false
      LOG_LEVEL=DEBUG
      MODEL_PATH=models/tfidf_lr_torch.pt
    """

    # Environment (dev/prod/etc.)
    ENV: str = os.getenv("ENV", "dev")

    # Logging
    ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() in (
        "1", "true", "yes", "on",
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # 🔑 ML Model path (added so _resolve_model_path can find it)
    MODEL_PATH: str | None = os.getenv("MODEL_PATH")


# Instantiate settings
settings = Settings()
