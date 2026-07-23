"""create projects table + wire deferred jobs.project_id FK

Revision: 20260510_05
Down revision: 20260510_04
Created: 2026-05-10

Creates the `projects` table per abstractions-spec § 5.4. Project is the
structural pivot of the application — workspace bound to one Scene,
ownership root for everything that grows under it (Actions /
Visualizations / Notes / Exports — none of which exist yet).

Two things in one migration:
1. Create the `projects` table.
2. Add the FK on `jobs.project_id` → `projects.id` that Step 6b's jobs
   migration deliberately deferred (the comment in
   `20260510_03_create_jobs.py` notes the constraint would land with
   the projects table). ON DELETE CASCADE so deleting a project
   cleans up its action_run / project_export jobs (per § 6 cascade
   table). The column was already indexed in Step 6b, so no new index
   here.
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260510_05"
down_revision: Union[str, None] = "20260510_04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="projects_user_fk",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["scene_id"],
            ["scenes.id"],
            name="projects_scene_fk",
            ondelete="RESTRICT",
        ),
    )

    op.create_index("projects_user_id_idx", "projects", ["user_id"])
    op.create_index("projects_scene_id_idx", "projects", ["scene_id"])

    # Wire the deferred FK on jobs.project_id (Step 6b put the column +
    # partial index in but skipped the constraint because projects didn't
    # exist yet). CASCADE so action_run / project_export jobs are
    # cleaned up when their project is deleted.
    op.create_foreign_key(
        "jobs_project_fk",
        source_table="jobs",
        referent_table="projects",
        local_cols=["project_id"],
        remote_cols=["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("jobs_project_fk", "jobs", type_="foreignkey")
    op.drop_index("projects_scene_id_idx", table_name="projects")
    op.drop_index("projects_user_id_idx", table_name="projects")
    op.drop_table("projects")
