import sys
import os

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generation.generator import AsterRowAgent

def main():
    kb_path = "knowledge-base"
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
    
    print("==================================================")
    print("      Initializing Aster & Row Support Agent      ")
    print("==================================================")
    
    agent = AsterRowAgent(kb_path, debug=debug_mode)
    session_id = "cli-session-001"
    
    print(f"\nSession ID: {session_id} | Debug Mode: {debug_mode}")
    print("Commands: Type your question, 'clear' to reset session, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            if user_input.lower() == 'clear':
                agent.session_store.clear_session(session_id)
                print("🔄 Session memory cleared.\n")
                continue

            # Run turn via agent getting exact Phase 6 contract
            response = agent.run_turn(user_input, session_id=session_id)

            print(f"\nAgent: {response['answer']}")
            if response['sources']:
                print("\nSources:")
                for src in response['sources']:
                    print(f"  - {src}")
            print(f"\nHuman Handoff Required: {response['human_handoff']}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()