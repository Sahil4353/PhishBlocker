import os
from pathlib import Path


class Settings:
    """
    Lightweight settings with env overrides + model auto-discovery.

    - If MODEL_PATH is set in the environment, that is used directly.
    - Otherwise, we auto-detect the "best" model from MODELS_DIR.

    Auto-discovery:
      - Look in MODELS_DIR (default: 'models')
      - Consider files with extensions: .pt (Torch), .joblib (sklearn)
      - Pick the most recently modified file
      - Try to match a metrics file: <stem>.metrics.json
      - Derive MODEL_VERSION from the stem (or env override if provided)
    """

    # --- General app settings ---
    ENV: str = os.getenv("ENV", "dev")

    ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Model discovery base directory ---
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "models"))

    def __init__(self) -> None:
        # Env overrides take priority
        env_model_path = os.getenv("MODEL_PATH", "").strip()
        env_metrics_path = os.getenv("MODEL_METRICS_PATH", "").strip()
        env_version = os.getenv("MODEL_VERSION", "").strip()

        if env_model_path:
            # Use explicit env config
            self.MODEL_PATH = env_model_path
            self.MODEL_METRICS_PATH = env_metrics_path
            # If version not given, derive from filename stem
            self.MODEL_VERSION = env_version or Path(env_model_path).stem
        else:
            # Auto-detect from MODELS_DIR
            model_path, metrics_path, model_version = self._autodetect_model()
            self.MODEL_PATH = model_path
            self.MODEL_METRICS_PATH = metrics_path
            # Allow env MODEL_VERSION to still override auto-detected version
            self.MODEL_VERSION = env_version or model_version

    # ----------------------------------------------------------------------
    def _autodetect_model(self) -> tuple[str, str, str]:
        """
        Return (model_path, metrics_path, model_version).

        - Scans MODELS_DIR (non-recursive).
        - Considers files with extensions: .pt, .joblib
        - Picks newest by mtime.
        - Looks for matching <stem>.metrics.json.
        - If nothing is found, returns empty strings (heuristics-only mode).
        """

        if not self.MODELS_DIR.exists():
            # No models directory at all → heuristics only.
            return "", "", ""

        # Consider Torch and sklearn artifacts
        allowed_ext = {".pt", ".joblib"}

        model_files = sorted(
            [
                p
                for p in self.MODELS_DIR.iterdir()
                if p.is_file() and p.suffix in allowed_ext
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not model_files:
            # No model files → heuristics only.
            return "", "", ""

        # Pick newest candidate
        best_model = model_files[0]
        stem = best_model.stem

        # Try to find matching metrics JSON
        metrics_path = self.MODELS_DIR / f"{stem}.metrics.json"
        metrics_str = str(metrics_path) if metrics_path.exists() else ""

        version = stem  # simple, filename-based version

        return str(best_model), metrics_str, version


settings = Settings()
