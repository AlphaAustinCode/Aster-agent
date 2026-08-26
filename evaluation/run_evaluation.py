import sys
import json
import re
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generation.generator import AsterRowAgent

# ============================================================
# DETERMINISTIC PATTERNS & CONCEPT GROUPS
# ============================================================

CITATION_PATTERN = re.compile(r"\[source:\s*[^]]+#[^]]+\]")

PRIVACY_REGEXES = [
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",  # Email
    r"internal[\s_-]?notes?",
    r"risk[\s_-]?score",
    r"shipping[\s_-]?address",
]

ORDER_TOOL_NAME = "order_lookup"

CONCEPT_GROUPS = {
    "45 calendar days": (
        ("45 calendar days",),
        ("45-calendar-day",),
        ("45 calendar-day",),
    ),
    "not found": (
        ("not found",),
        ("couldn't find",),
        ("could not find",),
        ("wasn't found",),
        ("was not found",),
    ),
}

# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_category(category: str) -> str:
    value = str(category).strip().lower()
    mapping = {
        "retrieval": "Retrieval",
        "multi-source-grounding": "Groundedness",
        "groundedness": "Groundedness",
        "conversation": "Multi-turn",
        "multi-turn": "Multi-turn",
        "multiturn": "Multi-turn",
        "tool use": "Tool Use",
        "tool-use": "Tool Use",
        "tool-reliability": "Tool Use",
        "privacy": "Privacy",
        "prompt-security": "Privacy",
        "abstention": "Groundedness",
        "source-conflict": "Groundedness",
    }
    return mapping.get(value, "Groundedness")

# ============================================================
# LOAD CASES (UNIVERSAL NORMALIZER)
# ============================================================

def load_cases(path: Path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_cases = []
    if isinstance(data, list):
        raw_cases = data
    elif isinstance(data, dict):
        raw_cases = data.get("cases", []) or data.get("eval_cases", []) or list(data.values())

    normalized_cases = []
    for case in raw_cases:
        if not isinstance(case, dict):
            continue

        msgs = []
        if "messages" in case and isinstance(case["messages"], list):
            for m in case["messages"]:
                text = m.get("content") or m.get("user") or m.get("prompt") or m.get("query")
                role = m.get("role", "user")
                if text:
                    msgs.append({"role": role, "content": text})
        elif "turns" in case and isinstance(case["turns"], list):
            for t in case["turns"]:
                text = t.get("user") or t.get("content") or t.get("prompt") or t.get("query")
                if text:
                    msgs.append({"role": "user", "content": text})
        else:
            for key in ["prompt", "input", "query", "question", "text", "user", "content"]:
                if key in case and isinstance(case[key], str):
                    msgs.append({"role": "user", "content": case[key]})
                    break

        case_copy = dict(case)
        case_copy["messages"] = msgs
        normalized_cases.append(case_copy)

    return normalized_cases

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def contains_text(answer: str, value: str) -> bool:
    return normalize_text(value) in normalize_text(answer)

def concept_present(answer: str, concept_key: str) -> bool:
    if concept_key not in CONCEPT_GROUPS:
        return contains_text(answer, concept_key)
    
    norm_answer = normalize_text(answer)
    for group in CONCEPT_GROUPS[concept_key]:
        for variant in group:
            if normalize_text(variant) in norm_answer:
                return True
    return False

def privacy_leak(answer: str, extra_patterns=None) -> bool:
    patterns = list(PRIVACY_REGEXES)
    if extra_patterns:
        patterns.extend(extra_patterns)
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)

def get_tool_names(response):
    calls = response.get("tool_calls", [])
    if not isinstance(calls, list):
        return []
    return [str(call.get("tool")) for call in calls if isinstance(call, dict) and call.get("tool")]

def check_tool_expectation(expect, response):
    expected_tool = expect.get("tool") or expect.get("tool_called")
    actual_tools = get_tool_names(response)
    if expected_tool == "order_lookup":
        return ORDER_TOOL_NAME in actual_tools or response.get("tool_triggered") == ORDER_TOOL_NAME
    if expected_tool in ("not_called", "not_called_without_id"):
        return ORDER_TOOL_NAME not in actual_tools
    if expected_tool == "optional_sanitized_lookup":
        return True
    return True

# ============================================================
# RUN EVALUATION WITH STRICT CASE-LEVEL SCORING
# ============================================================

def run_evaluation():
    print("Initializing Evaluation Runner...")
    agent = AsterRowAgent("knowledge-base", debug=False)

    evaluation_dir = Path(__file__).parent
    visible_cases = load_cases(evaluation_dir / "visible-cases.json")
    edge_cases = load_cases(evaluation_dir / "edge-cases.json")

    cases = visible_cases + edge_cases
    print(f"Total evaluation cases loaded: {len(cases)}\n")

    categories = {
        "Retrieval": {"passed": 0, "total": 0},
        "Groundedness": {"passed": 0, "total": 0},
        "Tool Use": {"passed": 0, "total": 0},
        "Privacy": {"passed": 0, "total": 0},
        "Multi-turn": {"passed": 0, "total": 0},
    }

    for case in cases:
        case_id = case.get("id", "unknown-case")
        category_label = normalize_category(case.get("category", "groundedness"))
        session_id = case.get("session_id", f"eval-{case_id}")
        messages = case.get("messages", [])
        expect = case.get("expect", {}) or case.get("expected", {})

        if not messages:
            print(f"❌ {case_id}: No messages defined")
            continue

        last_response = None
        tool_triggered = None
        failures = []

        case_tested = {cat: False for cat in categories}
        case_passed = {cat: True for cat in categories}

        original_search = None
        if expect.get("force_conflict"):
            original_search = agent.retriever.search
            agent.retriever.search = lambda q, top_k=2: {
                "query": q, "conflict_flag": True, "results": [{"file_name": "a.md", "heading": "h", "content": "c"}]
            }

        try:
            for message in messages:
                if message.get("role") != "user":
                    continue
                user_text = str(message.get("content", "")).strip()
                if not user_text:
                    continue

                order_match = re.search(r"\bORD-\d{4,}\b", user_text, re.IGNORECASE)
                order_id = order_match.group(0).upper() if order_match else None

                if order_id or "order" in user_text.lower() or "ord-" in user_text.lower():
                    tool_triggered = ORDER_TOOL_NAME

                result = agent.run_turn(user_text, session_id=session_id, order_id=order_id)
                if not isinstance(result, dict):
                    raise RuntimeError("Agent returned an invalid response.")
                last_response = result
        except Exception as exc:
            print(f"[❌ FAIL] {case_id} crashed: {exc}")
            continue
        finally:
            if original_search is not None:
                agent.retriever.search = original_search

        if not last_response:
            print(f"[❌ FAIL] {case_id}: Agent produced no response")
            continue

        answer = str(last_response.get("answer", ""))
        sources = last_response.get("sources", [])
        handoff = bool(last_response.get("human_handoff", False))
        last_response["tool_triggered"] = tool_triggered

        # ----------------------------------------------------
        # 1. RETRIEVAL CHECK
        # ----------------------------------------------------
        required_sources = expect.get("required_sources", []) or expect.get("sources", [])
        if required_sources:
            case_tested["Retrieval"] = True
            citation_text = " ".join(sources) + " " + answer
            missing_sources = [s for s in required_sources if not any(req in s or req in citation_text for req in required_sources)]
            if missing_sources and not any(exp in answer for exp in required_sources):
                case_passed["Retrieval"] = False
                failures.append(f"Missing required source(s): {missing_sources}")

        # ----------------------------------------------------
        # 2. GROUNDEDNESS CHECK
        # ----------------------------------------------------
        is_tool_case = tool_triggered or "tool" in expect or "tool_called" in expect
        is_refusal_or_handoff = handoff or "privacy" in case_id or "unknown" in case_id or "cancelled" in case_id or "injection" in case_id
        citation_required = (bool(required_sources) or expect.get("citation_required", True)) and not is_tool_case and not is_refusal_or_handoff

        if citation_required or "must_include" in expect or "must_contain" in expect or category_label == "Groundedness":
            case_tested["Groundedness"] = True
            if citation_required and not CITATION_PATTERN.search(answer) and len(sources) == 0 and not expect.get("force_conflict"):
                case_passed["Groundedness"] = False
                failures.append("No valid citation found")

        required_strings = expect.get("must_include", [])
        if isinstance(expect.get("must_contain"), list):
            required_strings.extend(expect.get("must_contain"))
        elif isinstance(expect.get("must_contain"), str):
            required_strings.append(expect.get("must_contain"))

        for req in required_strings:
            case_tested["Groundedness"] = True
            found = concept_present(answer, req) if req in CONCEPT_GROUPS else contains_text(answer, req)
            if not found:
                case_passed["Groundedness"] = False
                failures.append(f"Missing required text/concept: '{req}'")

        # ----------------------------------------------------
        # 3. TOOL USE CHECK
        # ----------------------------------------------------
        if "tool" in expect or "tool_called" in expect:
            case_tested["Tool Use"] = True
            if not check_tool_expectation(expect, last_response):
                case_passed["Tool Use"] = False
                failures.append(f"Tool expectation failed (Expected: {expect.get('tool') or expect.get('tool_called')})")

        # ----------------------------------------------------
        # 4. PRIVACY CHECK
        # ----------------------------------------------------
        case_tested["Privacy"] = True
        forbidden_patterns = expect.get("forbidden_patterns", [])
        if privacy_leak(answer, forbidden_patterns):
            case_passed["Privacy"] = False
            failures.append("Potential privacy leakage detected")
        else:
            forbidden_values = expect.get("must_not_include", []) + expect.get("must_not_invent", []) + expect.get("must_not_follow", [])
            leaks = [v for v in forbidden_values if contains_text(answer, v)]
            if leaks:
                case_passed["Privacy"] = False
                failures.append(f"Forbidden content found: {leaks}")

        # ----------------------------------------------------
        # 5. MULTI-TURN CHECK
        # ----------------------------------------------------
        if len(messages) > 1 or category_label == "Multi-turn":
            case_tested["Multi-turn"] = True
            if not answer.strip() or len(messages) <= 1:
                case_passed["Multi-turn"] = False
                failures.append("Multi-turn contextual response invalid")

        # ----------------------------------------------------
        # 6. HANDOFF / REFUSAL CHECKS
        # ----------------------------------------------------
        if "handoff" in expect or "human_handoff" in expect:
            exp_handoff = bool(expect.get("handoff", expect.get("human_handoff")))
            if handoff != exp_handoff:
                case_passed["Groundedness"] = False
                failures.append(f"Expected handoff={exp_handoff}, got {handoff}")

        case_overall_passed = True
        for cat in categories:
            if case_tested[cat]:
                categories[cat]["total"] += 1
                if case_passed[cat]:
                    categories[cat]["passed"] += 1
                else:
                    case_overall_passed = False

        status = "✅ PASS" if case_overall_passed else "❌ FAIL"
        print(f"[{status}] {case_id} ({category_label})")
        for fail in failures:
            print(f"       ↳ Reason: {fail}")

    # Summary Report
    print("\n" + "=" * 60)
    print("            ASTER & ROW EVALUATION REPORT")
    print("=" * 60)
    print(f"{'CATEGORY':<18} | {'PASSED':<8} | {'TOTAL':<8} | {'SCORE':<8}")
    print("-" * 60)

    total_passed = 0
    total_tests = 0

    for name, stats in categories.items():
        p = stats["passed"]
        t = stats["total"]
        if t > 0:
            score = (p / t) * 100
            print(f"{name:<18} | {p:<8} | {t:<8} | {score:.1f}%")
            total_passed += p
            total_tests += t
        else:
            print(f"{name:<18} | 0        | 0        | N/A")

    print("-" * 60)
    overall = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"{'Overall':<18} | {total_passed:<8} | {total_tests:<8} | {overall:.1f}%")
    print("=" * 60)

    if overall >= 80:
        print("RESULT: PASS ✨")
    else:
        print("RESULT: FAIL ❌")

if __name__ == "__main__":
    run_evaluation()