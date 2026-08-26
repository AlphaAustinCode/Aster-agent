from src.rag.indexer import (
    generate_chunk_id,
    DocumentChunk,
    validate_safe_index
)

def test_chunk_id_is_deterministic():

    first = generate_chunk_id(
        "refund-policy",
        3,
        "Eligibility"
    )

    second = generate_chunk_id(
        "refund-policy",
        3,
        "Eligibility"
    )

    assert first == second


def test_different_versions_have_different_ids():

    v1 = generate_chunk_id(
        "refund-policy",
        1,
        "Eligibility"
    )

    v2 = generate_chunk_id(
        "refund-policy",
        2,
        "Eligibility"
    )

    assert v1 != v2


def test_safe_index_rejects_internal_chunk():

    chunk = DocumentChunk(
        chunk_id="abc123",
        file_name="internal.md",
        document_id="internal-policy",
        heading="General",
        content="Internal content",
        metadata={
            "status": "active",
            "audience": "internal"
        }
    )

    try:
        validate_safe_index([chunk])
        assert False
    except ValueError:
        assert True



def test_safe_index_accepts_valid_chunk():

    chunk = DocumentChunk(
        chunk_id="abc123",
        file_name="refund.md",
        document_id="refund-policy",
        heading="Eligibility",
        content="Customer refund information",
        metadata={
            "status": "active",
            "audience": "customer",
            "version": 3
        }
    )

    validate_safe_index([chunk])