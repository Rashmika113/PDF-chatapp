import pytest

from rag.embeddings import prefixes_for


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("intfloat/multilingual-e5-base", ("query: ", "passage: ")),
        ("intfloat/multilingual-e5-small", ("query: ", "passage: ")),
        ("BAAI/bge-small-en-v1.5", ("Represent this sentence for searching relevant passages: ", "")),
        ("BAAI/bge-m3", ("", "")),
    ],
)
def test_prefixes_match_model_family(model, expected):
    assert prefixes_for(model) == expected
