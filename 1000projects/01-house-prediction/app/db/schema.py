from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateColumn

from app.db.database import Base


def sync_missing_columns(engine: Engine) -> None:
    """Add ORM columns that are missing from existing tables (create_all is no-op then)."""
    inspector = inspect(engine)
    dialect = engine.dialect

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                add_clause = CreateColumn(column.copy()).compile(dialect=dialect)
                conn.execute(text(f"ALTER TABLE {table.name} ADD {add_clause}"))
