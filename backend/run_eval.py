from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app, TOOL_HANDLERS, find_product, find_order
from eval_cases import EVAL_CASES

test_client = TestClient(app)


def run_evaluation(case):
    actual_tool_calls = []

    def traced_product(product_id):
        actual_tool_calls.append(
            {
                "name": "get_product",
                "arguments": {"product_id": product_id},
            }
        )

        return find_product(product_id)

    def traced_order(order_id):
        actual_tool_calls.append(
            {
                "name": "get_order",
                "arguments": {"order_id": order_id},
            }
        )

        return find_order(order_id)

    with patch.dict(
        TOOL_HANDLERS,
        {
            "get_product": traced_product,
            "get_order": traced_order,
        },
    ):
        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {
                        "role": "user",
                        "content": case["user_message"],
                    }
                ]
            },
        )

        tool_selection_passed = actual_tool_calls == case["expected_tool_calls"]

        return {
            "name": case["name"],
            "status_code": response.status_code,
            "tool_selection_passed": tool_selection_passed,
            "answer": response.json().get("answer"),
        }


if __name__ == "__main__":
    results = []
    count = 0

    for case in EVAL_CASES:
        result = run_evaluation(case)
        results.append(result)

        if result["tool_selection_passed"]:
            count += 1

    pass_rate = (count / len(EVAL_CASES)) * 100

    print("\nAgent Evaluation Summary")
    print("------------------------")
    print(f"Cases evaluated: {len(results)}")
    print(f"Tool sequences passed: {count}")
    print(f"Tool-sequence pass rate: {pass_rate:.1f}%")
