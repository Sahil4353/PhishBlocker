from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Scan(Base):
    __tablename__ = "scans"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="ck_scans_confidence"
        ),
        Index("ix_scans_created_at", "created_at"),
        Index("ix_scans_label", "label"),
        Index("ix_scans_sender", "sender"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    sender: Mapped[str | None] = mapped_column(String(255), nullable=True)
    body_preview: Mapped[str | None] = mapped_column(String(500), nullable=True)
    raw: Mapped[str | None] = mapped_column(Text, nullable=True)

    label: Mapped[str] = mapped_column(String(32), nullable=False, default="safe")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    reasons: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Scan id={self.id} label={self.label} conf={self.confidence:.2f}>"
