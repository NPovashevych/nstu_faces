"""update media paths

Revision ID: 6ae5d5826924
Revises: 51e1817baea3
Create Date: 2026-05-28 13:59:42.519205

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6ae5d5826924'
down_revision: Union[str, Sequence[str], None] = '51e1817baea3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("media", sa.Column("mxf_path", sa.String(), nullable=True))
    op.add_column("media", sa.Column("mp4_path", sa.String(), nullable=True))
    op.add_column("media", sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=True))

    op.drop_index("ix_media_media_path", table_name="media")
    op.drop_column("media", "media_path")

    op.create_index("ix_media_mxf_path", "media", ["mxf_path"], unique=True)
    op.create_index("ix_media_mp4_path", "media", ["mp4_path"], unique=True)


def downgrade() -> None:
    op.add_column("media", sa.Column("media_path", sa.String(), nullable=True))

    op.drop_index("ix_media_mp4_path", table_name="media")
    op.drop_index("ix_media_mxf_path", table_name="media")

    op.drop_column("media", "recorded_at")
    op.drop_column("media", "mp4_path")
    op.drop_column("media", "mxf_path")

    op.create_index("ix_media_media_path", "media", ["media_path"], unique=True)
