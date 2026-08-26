import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Path to orders.json relative to the repository root
DEFAULT_ORDERS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "orders.json"

# Whitelist of customer-safe fields. Anything not listed here is discarded.
ALLOWED_FIELDS = {
    "order_id",
    "status",
    "items",
    "order_date",
    "total",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "delivery_date",
}

# Regex strictly matching valid Aster & Row order IDs (e.g., ORD-1007)
ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")


def sanitize_and_validate_order_id(raw_id: Optional[str]) -> Optional[str]:
    """
    Cleans and validates the order ID.
    Handles trimming, uppercase conversion, and flexible separators.
    
    Accepts: 'ord-1007', '  ORD-1007  ', 'ord 1007', 'ORD1007'
    Rejects: None, '', 'hello', '1007', 'ORD', 'ORD-XYZ'
    """
    if raw_id is None:
        return None

    cleaned = str(raw_id).strip().upper()
    if not cleaned:
        return None

    # Handle inputs like "ORD 1007" or "ORD1007" by normalizing to "ORD-1007"
    normalized = re.sub(r"^ORD[-\s]*(\d{4})$", r"ORD-\1", cleaned)

    if ORDER_ID_PATTERN.match(normalized):
        return normalized

    return None


def lookup_order(
    order_id: Optional[str],
    orders_file_path: Path = DEFAULT_ORDERS_PATH
) -> Dict[str, Any]:
    """
    Secure order lookup tool for the agent.
    
    1. Validates and normalizes order ID format.
    2. Reads data/orders.json without exposing the raw file to the LLM.
    3. Drops internal fields (email, address, internal notes, risk scores).
    4. Strips stale delivery estimates for cancelled or returned orders.
    5. Formats dates nicely (e.g., August 22, 2026) when present.
    """
    # 1. Validate order ID
    valid_id = sanitize_and_validate_order_id(order_id)
    if not valid_id:
        return {"error": f"Invalid order ID format: '{order_id}'. Expected format like 'ORD-1001'."}

    # 2. Load order database
    if not orders_file_path.exists():
        return {"error": "Internal error: Order database is currently unreachable."}

    try:
        with open(orders_file_path, "r", encoding="utf-8") as f:
            orders = json.load(f)
            
            if isinstance(orders, dict):
                orders = orders.get("orders", [])
                
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"Internal error reading order data: {str(e)}"}

    # 3. Locate matching order
    target_order = next((o for o in orders if o.get("order_id") == valid_id), None)
    if not target_order:
        return {"error": f"Order '{valid_id}' not found."}

    # 4. Data Masking (Strict Whitelist Approach)
    safe_record: Dict[str, Any] = {
        key: target_order[key]
        for key in ALLOWED_FIELDS
        if key in target_order
    }

    # 5. Status Consistency & Stale Field Pruning
    status = str(safe_record.get("status", "")).lower().strip()

    if status in {"cancelled", "returned", "failed", "canceled"}:
        # Terminated orders must never present stale delivery expectations
        safe_record.pop("estimated_delivery", None)
        safe_record.pop("delivery_date", None)
        safe_record.pop("carrier", None)
        safe_record.pop("tracking_number", None)
    else:
        # Format date nicely if possible (e.g., August 22, 2026)
        est_delivery = safe_record.get("estimated_delivery")
        if est_delivery:
            try:
                dt = datetime.fromisoformat(est_delivery)
                safe_record["estimated_delivery"] = dt.strftime("%B %d, %Y")
            except Exception:
                pass  # Keep original string if parsing fails
        elif status == "processing":
            # Ensure processing orders do not invent missing delivery dates
            safe_record["estimated_delivery"] = None

    return safe_record