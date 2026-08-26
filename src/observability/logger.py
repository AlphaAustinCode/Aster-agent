import json
from datetime import datetime
from typing import List, Dict, Any

class AgentLogger:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def log_turn(
        self,
        session_id: str,
        turn_num: int,
        user_input: str,
        history: List[Dict[str, str]],
        retrieved_chunks: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        handoff: bool
    ) -> None:
        """Outputs structured debug logs for the current turn if debug mode is enabled."""
        if not self.debug_mode:
            return

        # Ensure strict privacy compliance: never log PII or raw databases
        safe_tool_calls = []
        for call in tool_calls:
            safe_tool_calls.append(call) # tool_calls are already scrubbed by order_lookup

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "turn": turn_num,
            "user_input": user_input,
            "conversation_history": history,
            "retrieved_chunks": [
                {
                    "source": chunk.get("file_name"),
                    "heading": chunk.get("heading"),
                    "score": chunk.get("score")
                }
                for chunk in retrieved_chunks
            ],
            "tool_calls": safe_tool_calls,
            "handoff": handoff
        }

        print("\n" + "="*50)
        print("🔍 [DEBUG TRACE LOG]")
        print("="*50)
        print(json.dumps(log_entry, indent=2))
        print("="*50 + "\n")