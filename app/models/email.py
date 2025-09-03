# app/models/email.py
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from sqlalchemy import JSON, CheckConstraint, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .scan import Scan


class Email(Base):
    """
    Normalized email entity (headers + bodies + parsed fields).
    Scans reference this row via Email.id; we keep common header/body pieces here
    to avoid duplication across multiple scans of the same message.
    """

    __tablename__ = "emails"
    __table_args__ = (
        Index("ix_emails_message_id", "message_id"),
        Index("ix_emails_sender", "sender"),
        Index("ix_emails_timestamp", "timestamp"),
        # data hygiene: booleans as 0/1; attachments count non-negative
        CheckConstraint("has_text IN (0,1)", name="ck_emails_has_text_bool"),
        CheckConstraint("has_html IN (0,1)", name="ck_emails_has_html_bool"),
        CheckConstraint("attachments_count >= 0", name="ck_emails_attachments_nonneg"),
    )

    # Short stable hash from ingesters as PK (<=32 chars)
    id: Mapped[str] = mapped_column(String(32), primary_key=True)

    # Provenance
    source: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # gmail|outlook|enron|...

    # Core RFC-822 headers
    message_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    subject: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sender: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    recipients: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # To;Cc;Bcc joined
    reply_to: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    return_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Bodies
    body_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    html_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_text: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0/1
    has_html: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0/1

    # Links & attachments
    urls: Mapped[Optional[List | Dict]] = mapped_column(
        JSON, nullable=True
    )  # JSON array
    attachments: Mapped[Optional[List | Dict]] = mapped_column(
        JSON, nullable=True
    )  # JSON array
    attachments_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # List and auth headers
    list_unsubscribe: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    spf_result: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )  # pass|fail|none|...
    dkim_result: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    dmarc_result: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    received_ips: Mapped[Optional[List | Dict]] = mapped_column(
        JSON, nullable=True
    )  # JSON array

    # Optional raw storage/pointers
    raw_headers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_eml_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Backref to scans
    scans: Mapped[list["Scan"]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Email id={self.id} src={self.source} sender={self.sender!r}>"
