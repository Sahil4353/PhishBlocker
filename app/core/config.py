import os
from pathlib import Path


class Settings:
    """
    Auto-discovers the latest ML model inside `models/`.

    Expected files:
      *.joblib
      *.metrics.json

    It will pick the newest modified *.joblib file
    and try to load matching metrics with same stem.
    """

    ENV: str = os.getenv("ENV", "dev")
    ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    MODELS_DIR = Path("models")

    def __init__(self):
        self.MODEL_PATH, self.MODEL_METRICS_PATH, self.MODEL_VERSION = (
            self._autodetect_model()
        )

    # ----------------------------------------------------------------------
    def _autodetect_model(self):
        """
        Return (model_path, metrics_path, model_version)
        """

        # list all *.joblib files directly in models root
        model_files = sorted(
            [p for p in self.MODELS_DIR.glob("*.joblib") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not model_files:
            # fallback — same behavior as your current default
            default_model = self.MODELS_DIR / "tfidf_lr_small_l2.joblib"
            default_metrics = self.MODELS_DIR / "tfidf_lr_small_l2.metrics.json"
            return str(default_model), str(default_metrics), "fallback_default"

        # pick newest model
        best_model = model_files[0]
        stem = best_model.stem

        # matching metrics
        metrics = self.MODELS_DIR / f"{stem}.metrics.json"
        if not metrics.exists():
            metrics = None

        version = stem  # simple version naming

        return str(best_model), str(metrics) if metrics else "", version


settings = Settings()
