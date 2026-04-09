"""Add callout graph tables: sheet_hyperlinks, callout_links

Revision ID: 002
Revises: 001
Create Date: 2026-04-09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sheet_hyperlinks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_sheet_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sheets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_sheet_number", sa.String(50), nullable=False),
        sa.Column("context_text", sa.String(255)),
        sa.Column("bbox_x", sa.Float),
        sa.Column("bbox_y", sa.Float),
        sa.Column("bbox_w", sa.Float),
        sa.Column("bbox_h", sa.Float),
    )
    op.create_index("idx_hyperlinks_source", "sheet_hyperlinks", ["source_sheet_id"])
    op.create_index("idx_hyperlinks_target", "sheet_hyperlinks", ["target_sheet_number"])

    op.create_table(
        "callout_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_sheet_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sheets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_sheet_number", sa.String(50), nullable=False),
        sa.Column("target_detail_number", sa.String(10)),
        sa.Column("bbox_x", sa.Float),
        sa.Column("bbox_y", sa.Float),
        sa.Column("bbox_w", sa.Float),
        sa.Column("bbox_h", sa.Float),
    )
    op.create_index("idx_callout_source", "callout_links", ["source_sheet_id"])


def downgrade() -> None:
    op.drop_table("callout_links")
    op.drop_table("sheet_hyperlinks")
