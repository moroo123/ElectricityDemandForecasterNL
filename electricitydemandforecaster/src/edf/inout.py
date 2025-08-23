from pathlib import Path
import sqlite3
import pandas as pd


def read_dataframe_from_sql(db_path: str, table_name: str, column_names: list[str], timestamp_col: str, timestamp_format: str, last_rows: int | None = None):
    """Read a dataframe from a SQL database.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to read from.
        column_names (list[str]): List of column names to retrieve.
        timestamp_col (str): Name of the timestamp column.
        timestamp_format (str): Format of the timestamp column.

    Returns:
        pd.DataFrame: The resulting dataframe.
    """

    connection = sqlite3.connect(db_path)

    if last_rows is not None:
        df = pd.read_sql(
            f"SELECT {', '.join(column_names)} FROM {table_name} ORDER BY {timestamp_col} DESC LIMIT {last_rows}", connection)[::-1]
    else:
        df = pd.read_sql(
            f"SELECT {', '.join(column_names)} FROM {table_name}", connection)

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], format=timestamp_format)

    df.index = df[timestamp_col]
    df = df.drop(columns=[timestamp_col])

    df.dropna(inplace=True)

    return df


def project_root() -> Path:
    """Find repo root by walking up until we see pyproject.toml or .git."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def resolve_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)
