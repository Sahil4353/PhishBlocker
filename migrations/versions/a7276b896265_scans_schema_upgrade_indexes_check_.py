"""scans schema upgrade (indexes, check, reasons Text, created_at default)

Revision ID: a7276b896265
Revises:
Create Date: 2025-08-12 18:53:02.673539

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7276b896265"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The batch context makes SQLite recreate the table safely
    with op.batch_alter_table("scans") as batch:
        # reasons: String(500) -> Text (if already Text, this is a no-op)
        batch.alter_column(
            "reasons",
            existing_type=sa.String(length=500),
            type_=sa.Text(),
            existing_nullable=True,
        )

        # created_at: add server default CURRENT_TIMESTAMP and tz-aware type
        batch.alter_column(
            "created_at",
            existing_type=sa.DateTime(timezone=False),
            type_=sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            existing_nullable=False,
        )

        # add check constraint for confidence in [0,1] (if it doesn't exist)
        batch.create_check_constraint(
            "ck_scans_confidence",
            "confidence >= 0.0 AND confidence <= 1.0",
        )

        # helpful indexes (if missing)
        batch.create_index("ix_scans_created_at", ["created_at"], unique=False)
        batch.create_index("ix_scans_label", ["label"], unique=False)
        batch.create_index("ix_scans_sender", ["sender"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("scans") as batch:
        # reverse indexes
        batch.drop_index("ix_scans_sender")
        batch.drop_index("ix_scans_label")
        batch.drop_index("ix_scans_created_at")

        # drop check constraint
        batch.drop_constraint("ck_scans_confidence", type_="check")

        # revert created_at changes
        batch.alter_column(
            "created_at",
            existing_type=sa.DateTime(timezone=True),
            type_=sa.DateTime(timezone=False),
            server_default=None,
            existing_nullable=False,
        )

        # revert reasons to String(500)
        batch.alter_column(
            "reasons",
            existing_type=sa.Text(),
            type_=sa.String(length=500),
            existing_nullable=True,
        )
