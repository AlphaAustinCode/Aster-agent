import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.generator import AsterRowAgent
from src.generation.guardrails import ABSTENTION_STRING, CONFLICT_STRING

def run_edge_case_tests():
    print("Loading Agent...")
    agent = AsterRowAgent("knowledge-base")
    
    print("\n==================================================")
    print("TEST 1: INSUFFICIENT INFORMATION (Out of Domain)")
    print("==================================================")
    # Asking a question completely unrelated to Aster & Row's corpus
    query_1 = "How do I bake a chocolate cake?"
    print(f"User: {query_1}")
    
    response_1 = agent.process_message(query_1)
    print(f"\nAgent:\n{response_1}")
    
    if ABSTENTION_STRING in response_1:
        print("\n✅ PASS: Agent correctly escalated due to insufficient information.")
    else:
        print("\n❌ FAIL: Agent hallucinated or failed to use exact abstention string.")

    print("\n==================================================")
    print("TEST 2: CONFLICTING ACTIVE POLICIES")
    print("==================================================")
    # We will temporarily mock the retriever's search method to force a conflict flag,
    # proving that our generator.py properly intercepts it before asking the LLM.
    original_search = agent.retriever.search
    
    def mock_conflict_search(query, top_k):
        return {
            "query": query,
            "conflict_flag": True, # FORCING THE CONFLICT FLAG
            "results": [
                {"file_name": "policy_A.md", "heading": "Rules", "content": "Rule is X", "metadata": {}},
                {"file_name": "policy_B.md", "heading": "Rules", "content": "Rule is Y", "metadata": {}}
            ]
        }
    
    agent.retriever.search = mock_conflict_search
    
    query_2 = "What are the rules?"
    print(f"User: {query_2} (Retriever flagged conflict = True)")
    
    response_2 = agent.process_message(query_2)
    print(f"\nAgent:\n{response_2}")
    
    if CONFLICT_STRING in response_2:
        print("\n✅ PASS: Agent correctly intercepted conflict and escalated.")
    else:
        print("\n❌ FAIL: Agent ignored conflict flag.")
        
    # Restore retriever
    agent.retriever.search = original_search

if __name__ == "__main__":
    run_edge_case_tests()