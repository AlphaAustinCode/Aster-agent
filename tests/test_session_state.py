import pytest
from src.generation.generator import AsterRowAgent

@pytest.fixture
def agent():
    """Initializes a fresh agent for each test."""
    return AsterRowAgent("knowledge-base")

def test_basic_memory_and_pronoun_resolution(agent):
    """Test 1 & 4: Order context carries over ('it' resolves to ORD-1001)"""
    session = "session_a"
    
    # Turn 1
    resp1 = agent.process_message("Where is ORD-1001?", session_id=session)
    assert "shipped" in resp1.lower()
    
    # Turn 2: Pronoun resolution
    resp2 = agent.process_message("When will it arrive?", session_id=session)
    assert "2026-08-30" in resp2  # The date from our mock data for ORD-1001
    assert "ORD-1001" in resp2.upper()

def test_session_isolation(agent):
    """Test 2: Session B must not inherit ORD-1001 from Session A."""
    agent.process_message("Where is ORD-1001?", session_id="session_a")
    
    # Session B asks a vague question without prior context
    resp_b = agent.process_message("When will it arrive?", session_id="session_b")
    
    # It should NOT mention ORD-1001 or its delivery date
    assert "ORD-1001" not in resp_b
    assert "2026-08-30" not in resp_b
    # Should ask for clarification
    assert "order number" in resp_b.lower() or "support" in resp_b.lower()

def test_policy_follow_up(agent):
    """Test 3: 'What about Canada?' resolves in context of international shipping."""
    session = "session_policy"
    
    # Turn 1
    resp1 = agent.process_message("Do you ship internationally?", session_id=session)
    assert "[source:" in resp1 # Should have citation
    
    # Turn 2: Contextual RAG
    resp2 = agent.process_message("What about Canada?", session_id=session)
    # The retriever should have pulled the international shipping policy for Canada
    assert "canada" in resp2.lower()
    assert "[source: 06-international-shipping.md" in resp2

def test_multiple_simultaneous_sessions(agent):
    """Test 5: Concurrent sessions maintain their own state."""
    # Seed Session A with ORD-1001
    agent.process_message("Where is ORD-1001?", session_id="session_a")
    
    # Seed Session B with ORD-1004 (Cancelled order in our mock data)
    agent.process_message("Where is ORD-1004?", session_id="session_b")
    
    # Query both simultaneously
    resp_a = agent.process_message("What's the status?", session_id="session_a")
    resp_b = agent.process_message("What's the status?", session_id="session_b")
    
    assert "shipped" in resp_a.lower()
    assert "cancelled" in resp_b.lower()

def test_unknown_session_is_empty(agent):
    """Test 6: Querying a completely new session_id starts fresh."""
    resp = agent.process_message("What did I just say?", session_id="unknown_session")
    history = agent.session_store.get_messages("unknown_session")
    
    # History should only contain the message just sent and its response
    assert len(history) == 2