from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .scan import Scan

VALID_LABELS = ("safe", "spam", "phishing")

class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("user_label IN ('safe','spam','phishing')", name="ck_feedback_user_label"),
        Index("ix_feedback_scan_id", "scan_id"),
        Index("ix_feedback_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(Integer, ForeignKey("scans.id", ondelete="CASCADE"), nullable=False)
    user_label: Mapped[str] = mapped_column(String(16), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "ui" | "api" | etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # optional: backref (no cascade needed here)
    scan: Mapped[Scan] = relationship()
