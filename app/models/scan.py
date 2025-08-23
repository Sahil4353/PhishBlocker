from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

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
    __tablename__ = "scans"
    __table_args__ = (
        # Name kept simple so it fits common Alembic naming conventions
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="confidence"),
        Index("ix_scans_created_at", "created_at"),
        Index("ix_scans_label", "label"),
        Index("ix_scans_sender", "sender"),
        Index("ix_scans_email_id", "email_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Link to normalized email content
    email_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("emails.id", ondelete="CASCADE"), nullable=True
    )
    email: Mapped["Email"] = relationship(back_populates="scans")

    # Cached metadata for fast list views (avoids join on common pages)
    subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    sender: Mapped[str | None] = mapped_column(String(255), nullable=True)
    body_preview: Mapped[str | None] = mapped_column(String(500), nullable=True)
    raw: Mapped[str | None] = mapped_column(Text, nullable=True)  # optional raw blob

    # Classification result
    label: Mapped[str] = mapped_column(
        String(32), nullable=False, default="safe"
    )  # safe|spam|phishing
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    reasons: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON/text of top indicators
    header_flags: Mapped[dict | list | None] = mapped_column(
        JSON, nullable=True
    )  # heuristics booleans, etc.
    model_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # e.g., 2025-08-24.tfidf_lr.v1
    direction: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # inbox|sent|upload

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Scan id={self.id} label={self.label} "
            f"conf={self.confidence:.2f} email_id={self.email_id}>"
        )
