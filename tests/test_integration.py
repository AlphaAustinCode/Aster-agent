from pathlib import Path
from src.rag.indexer import (
    build_safe_index,
    validate_safe_index,
    validate_index_integrity
)

# Define the missing path here!
KB_PATH = Path("knowledge-base")

def test_real_index_passes_integrity_check():

    chunks = build_safe_index(
        str(KB_PATH)
    )

    validate_index_integrity(chunks)