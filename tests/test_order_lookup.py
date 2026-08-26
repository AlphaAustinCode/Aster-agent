import json
import tempfile
from pathlib import Path
import pytest

from src.tools.order_lookup import lookup_order, sanitize_and_validate_order_id

# Mock database testing all edge cases: PII, internal notes, cancelled status with stale dates
MOCK_ORDERS = [
    {
        "order_id": "ORD-1001",
        "customer_email": "customer1@example.com",
        "shipping_address": "123 Maple St, Seattle, WA",
        "internal_notes": "VIP customer, handle with care",
        "risk_score": 0.05,
        "status": "shipped",
        "items": [{"name": "Breeze Tumbler", "quantity": 1}],
        "total": 38.00,
        "carrier": "USPS",
        "tracking_number": "9400111899223100000000",
        "estimated_delivery": "2026-08-30"
    },
    {
        "order_id": "ORD-1004",
        "customer_email": "customer4@example.com",
        "shipping_address": "456 Oak Ave, Portland, OR",
        "internal_notes": "Cancelled per customer request via email",
        "risk_score": 0.91,
        "status": "cancelled",
        "items": [{"name": "Canvas Travel Tote", "quantity": 1}],
        "total": 120.00,
        "carrier": "FedEx",
        "tracking_number": "794612345678",
        "estimated_delivery": "2026-08-28"  # Stale date that must be stripped
    },
    {
        "order_id": "ORD-1005",
        "customer_email": "customer5@example.com",
        "shipping_address": "789 Pine Rd, Austin, TX",
        "internal_notes": "Returned - item defective",
        "risk_score": 0.12,
        "status": "returned",
        "items": [{"name": "Everyday Water Bottle", "quantity": 2}],
        "total": 64.00,
        "carrier": "UPS",
        "tracking_number": "1Z9999999999999999",
        "delivery_date": "2026-08-15"       # Stale date that must be stripped
    }
]


@pytest.fixture
def mock_orders_file():
    """Creates a temporary orders.json file for deterministic testing."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(MOCK_ORDERS, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


# --- Input Sanitization Tests ---

@pytest.mark.parametrize("raw_input, expected", [
    ("ORD-1007", "ORD-1007"),
    ("ord-1007", "ORD-1007"),
    ("  ORD-1007  ", "ORD-1007"),
    ("  ord-1007\n", "ORD-1007"),
    ("ord 1007", "ORD-1007"),
    ("ORD1007", "ORD-1007"),
])
def test_valid_order_id_normalization(raw_input, expected):
    assert sanitize_and_validate_order_id(raw_input) == expected


@pytest.mark.parametrize("invalid_input", [
    "",
    "   ",
    "hello",
    "1007",
    "ORD",
    "ORD-XYZ",
    "ORD-12345",
    None
])
def test_invalid_order_id_rejection(invalid_input):
    assert sanitize_and_validate_order_id(invalid_input) is None
    result = lookup_order(invalid_input)
    assert "error" in result
    assert "Invalid order ID" in result["error"]


# --- Privacy & Data Masking Tests ---

def test_privacy_fields_stripped(mock_orders_file):
    result = lookup_order("ord-1001", orders_file_path=mock_orders_file)
    
    assert "error" not in result
    assert result["order_id"] == "ORD-1001"
    assert result["status"] == "shipped"
    
    # Assert private fields are strictly absent
    forbidden_fields = ["customer_email", "shipping_address", "internal_notes", "risk_score"]
    for field in forbidden_fields:
        assert field not in result, f"Security Violation: '{field}' leaked to tool output"


# --- Status Consistency & Stale Field Tests ---

def test_cancelled_order_strips_stale_delivery_fields(mock_orders_file):
    result = lookup_order("ORD-1004", orders_file_path=mock_orders_file)
    
    assert result["status"] == "cancelled"
    assert "estimated_delivery" not in result
    assert "carrier" not in result
    assert "tracking_number" not in result


def test_returned_order_strips_stale_delivery_fields(mock_orders_file):
    result = lookup_order("ORD-1005", orders_file_path=mock_orders_file)
    
    assert result["status"] == "returned"
    assert "delivery_date" not in result
    assert "carrier" not in result
    assert "tracking_number" not in result


def test_unknown_order_id(mock_orders_file):
    result = lookup_order("ORD-9999", orders_file_path=mock_orders_file)
    assert "error" in result
    assert "not found" in result["error"].lower()