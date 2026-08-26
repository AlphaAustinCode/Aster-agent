from datetime import date

from src.rag.indexer import (
    validate_metadata,
    is_current_document
)


def test_valid_metadata():

    metadata = {
        "document_id": "refund-policy",
        "title": "Refund Policy",
        "status": "active",
        "audience": "customer",
        "version": 2,
        "effective_date": "2026-01-01"
    }

    result = validate_metadata(metadata)

    assert result.document_id == "refund-policy"
    assert result.version == 2
    assert result.status == "active"


def test_invalid_status():

    metadata = {
        "document_id": "refund-policy",
        "status": "banana",
        "audience": "customer"
    }

    try:
        validate_metadata(metadata)
        assert False
    except ValueError:
        assert True


def test_future_document_is_not_current():

    metadata = {
        "document_id": "future-policy",
        "status": "active",
        "audience": "customer",
        "version": 1,
        "effective_date": "2026-09-01"
    }

    document = validate_metadata(metadata)

    assert is_current_document(
        document,
        date(2026, 8, 25)
    ) is False


def test_expired_document_is_not_current():

    metadata = {
        "document_id": "old-policy",
        "status": "active",
        "audience": "customer",
        "version": 1,
        "effective_date": "2026-01-01",
        "expires_at": "2026-07-31"
    }

    document = validate_metadata(metadata)

    assert is_current_document(
        document,
        date(2026, 8, 25)
    ) is False


def test_current_document_is_current():

    metadata = {
        "document_id": "current-policy",
        "status": "active",
        "audience": "customer",
        "version": 2,
        "effective_date": "2026-01-01"
    }

    document = validate_metadata(metadata)

    assert is_current_document(
        document,
        date(2026, 8, 25)
    ) is True