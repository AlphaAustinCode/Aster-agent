import pytest

from src.rag.indexer import (
    DocumentMetadata,
    resolve_precedence
)

def test_highest_version_wins():

    documents = [
        DocumentMetadata(
            document_id="refund-policy",
            title="Refund Policy V1",
            status="active",
            audience="customer",
            version=1
        ),
        DocumentMetadata(
            document_id="refund-policy",
            title="Refund Policy V2",
            status="active",
            audience="customer",
            version=2
        ),
        DocumentMetadata(
            document_id="refund-policy",
            title="Refund Policy V3",
            status="active",
            audience="customer",
            version=3
        )
    ]

    result = resolve_precedence(documents)

    assert len(result) == 1
    assert result[0].version == 3


def test_different_documents_are_preserved():

    documents = [
        DocumentMetadata(
            document_id="refund-policy",
            title="Refund Policy",
            status="active",
            audience="customer",
            version=3
        ),
        DocumentMetadata(
            document_id="shipping-policy",
            title="Shipping Policy",
            status="active",
            audience="customer",
            version=2
        )
    ]

    result = resolve_precedence(documents)

    assert len(result) == 2



def test_duplicate_highest_version_raises_conflict():

    documents = [
        DocumentMetadata(
            document_id="refund-policy",
            title="Refund Policy A",
            status="active",
            audience="customer",
            version=3,
            file_name="refund-a.md"
        ),
        DocumentMetadata(
            document_id="refund-policy",
            title="Refund Policy B",
            status="active",
            audience="customer",
            version=3,
            file_name="refund-b.md"
        )
    ]

    with pytest.raises(ValueError):
        resolve_precedence(documents)