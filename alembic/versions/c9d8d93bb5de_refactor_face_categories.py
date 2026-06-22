"""refactor face categories

Revision ID: c9d8d93bb5de
Revises: 6ae5d5826924
Create Date: 2026-05-28 16:52:02.706419

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "8a7f3c2d9b10"
down_revision: Union[str, Sequence[str], None] = "51e1817baea3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


face_category_enum = postgresql.ENUM(
    "real_identifiable",
    "real_unidentifiable",
    "non_human",
    "artificial_human",
    "ai_generated",
    "uncertain",
    name="face_category",
)


def upgrade() -> None:
    face_category_enum.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "face",
        sa.Column(
            "category",
            face_category_enum,
            nullable=False,
            server_default="uncertain",
        ),
    )
    op.add_column("face", sa.Column("category_score", sa.Float(), nullable=True))
    op.add_column(
        "face",
        sa.Column("analysis", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index(op.f("ix_face_category"), "face", ["category"], unique=False)
    op.create_index(op.f("ix_face_category_score"), "face", ["category_score"], unique=False)

    op.drop_index(op.f("ix_face_suspicion_reason"), table_name="face")
    op.drop_index(op.f("ix_face_is_suspicious"), table_name="face")
    op.drop_index(op.f("ix_face_clip_category"), table_name="face")

    op.drop_column("face", "clip_scores")
    op.drop_column("face", "clip_score")
    op.drop_column("face", "clip_category")
    op.drop_column("face", "suspicion_reason")
    op.drop_column("face", "is_suspicious")


def downgrade() -> None:
    op.add_column(
        "face",
        sa.Column(
            "is_suspicious",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column("face", sa.Column("suspicion_reason", sa.String(), nullable=True))
    op.add_column("face", sa.Column("clip_category", sa.String(), nullable=True))
    op.add_column("face", sa.Column("clip_score", sa.Float(), nullable=True))
    op.add_column(
        "face",
        sa.Column("clip_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index(op.f("ix_face_clip_category"), "face", ["clip_category"], unique=False)
    op.create_index(op.f("ix_face_is_suspicious"), "face", ["is_suspicious"], unique=False)
    op.create_index(op.f("ix_face_suspicion_reason"), "face", ["suspicion_reason"], unique=False)

    op.drop_index(op.f("ix_face_category_score"), table_name="face")
    op.drop_index(op.f("ix_face_category"), table_name="face")

    op.drop_column("face", "analysis")
    op.drop_column("face", "category_score")
    op.drop_column("face", "category")

    face_category_enum.drop(op.get_bind(), checkfirst=True)
