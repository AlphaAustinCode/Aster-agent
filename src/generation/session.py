from typing import List, Dict

class SessionStore:
    """In-memory session store to maintain conversational context per session ID."""
    
    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves the history for a session. Creates an empty session if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Appends a message to the session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})

    def clear_session(self, session_id: str) -> None:
        """Clears a session completely, ensuring no residual data."""
        if session_id in self._sessions:
            self._sessions[session_id] = []