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


settings = Settings()
