"""merge heads

Revision ID: 0ba800e22028
Revises: 6ae5d5826924, 8a7f3c2d9b10
Create Date: 2026-05-28 17:22:45.164948

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0ba800e22028'
down_revision: Union[str, Sequence[str], None] = ('6ae5d5826924', '8a7f3c2d9b10')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
