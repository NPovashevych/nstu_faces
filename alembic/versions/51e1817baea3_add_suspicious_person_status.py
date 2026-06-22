"""add suspicious person status

Revision ID: 51e1817baea3
Revises: b6dcc84275ff
Create Date: 2026-05-26 17:01:44.442008

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '51e1817baea3'
down_revision: Union[str, Sequence[str], None] = 'b6dcc84275ff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE person_status ADD VALUE IF NOT EXISTS 'suspicious'")


def downgrade() -> None:
    pass
