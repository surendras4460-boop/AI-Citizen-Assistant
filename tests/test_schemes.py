"""Unit tests for schemes data loading and transformation."""
import pandas as pd
from src.schemes_loader import load_schemes_df, schemes_to_chunks, clean_text


def test_clean_text_helper():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("  Scholarship  ") == "Scholarship"


def test_load_schemes_dataframe():
    df = load_schemes_df("schemes.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "scheme_name" in df.columns
    assert "schemeCategory" in df.columns
    assert "documents" in df.columns


def test_schemes_to_chunks_conversion():
    df = load_schemes_df("schemes.csv")
    chunks = schemes_to_chunks(df)
    assert len(chunks) >= len(df)
    for c in chunks:
        assert c.meta["kind"] == "scheme"
        assert len(c.text) > 20
