"""Initial schema: projects, sheets, ocr_words

Revision ID: 001
Revises:
Create Date: 2026-04-09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "sheets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sheet_number", sa.String(50), nullable=False),
        sa.Column("sheet_title", sa.String(255), nullable=False),
        sa.Column("discipline", sa.String(50), nullable=False),
        sa.Column("revision_date", sa.String(20)),
        sa.Column("scale", sa.String(50)),
        sa.Column("image_path", sa.Text, nullable=False),
        sa.Column("dzi_path", sa.Text, nullable=False),
        sa.Column("ocr_result_path", sa.Text),
        sa.Column("qdrant_point_id", sa.BigInteger),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("project_id", "sheet_number"),
    )
    op.create_index("idx_sheets_project", "sheets", ["project_id"])
    op.create_index("idx_sheets_discipline", "sheets", ["discipline"])

    op.create_table(
        "ocr_words",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("sheet_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sheets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("poly_x1", sa.Float),
        sa.Column("poly_y1", sa.Float),
        sa.Column("poly_x2", sa.Float),
        sa.Column("poly_y2", sa.Float),
        sa.Column("poly_x3", sa.Float),
        sa.Column("poly_y3", sa.Float),
        sa.Column("poly_x4", sa.Float),
        sa.Column("poly_y4", sa.Float),
    )
    op.create_index("idx_ocr_words_sheet", "ocr_words", ["sheet_id"])
    op.execute("CREATE INDEX idx_ocr_words_content_trgm ON ocr_words USING GIN (content gin_trgm_ops)")


def downgrade() -> None:
    op.drop_table("ocr_words")
    op.drop_table("sheets")
    op.drop_table("projects")
