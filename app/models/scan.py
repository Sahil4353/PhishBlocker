# app/models/scan.py
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .email import Email


class Scan(Base):
    """
    A single classification result for an email (or pasted text).
    Stores a fast "summary row" for list/detail views, plus optional
    structured outputs (probs/details).
    """

    __tablename__ = "scans"
    __table_args__ = (
        # Confidence is a probability [0, 1]
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="confidence"),
        Index("ix_scans_created_at", "created_at"),
        Index("ix_scans_label", "label"),
        Index("ix_scans_sender", "sender"),
        Index("ix_scans_email_id", "email_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Link to normalized email content
    email_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("emails.id", ondelete="CASCADE"), nullable=True
    )
    email: Mapped["Email"] = relationship(back_populates="scans")

    # Cached metadata for fast list views (avoids join on common pages)
    subject: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sender: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    body_preview: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # optional raw blob

    # Classification result
    label: Mapped[str] = mapped_column(
        String(32), nullable=False, default="safe"
    )  # safe|spam|phishing (or custom)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    reasons: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # comma-joined tokens (human-readable)

    # Heuristic flags from parser/header analysis
    header_flags: Mapped[Optional[Dict | List]] = mapped_column(JSON, nullable=True)

    # Structured model outputs (optional, schema has these as generic names)
    probs: Mapped[Optional[Dict | List]] = mapped_column(
        JSON, nullable=True
    )  # e.g., {"safe":0.98,"spam":0.01,"phishing":0.01}
    details: Mapped[Optional[Dict | List]] = mapped_column(
        JSON, nullable=True
    )  # e.g., {"reasons":[{"token":"verify","weight":1.2}], ...}

    model_version: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # e.g., "2025-08-30.tfidf_lr.v1"
    direction: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # inbox|sent|upload

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    # ---- Compatibility aliases (so services can write details_json / probs_json) ----
    @property
    def details_json(self) -> Optional[Dict | List]:
        return self.details

    @details_json.setter
    def details_json(self, value: Optional[Dict | List]) -> None:
        self.details = value

    @property
    def probs_json(self) -> Optional[Dict | List]:
        return self.probs

    @probs_json.setter
    def probs_json(self, value: Optional[Dict | List]) -> None:
        self.probs = value

    def __repr__(self) -> str:
        return f"<Scan id={self.id} label={self.label} conf={self.confidence:.2f} email_id={self.email_id}>"
