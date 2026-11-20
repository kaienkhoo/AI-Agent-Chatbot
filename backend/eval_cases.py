EVAL_CASES = [
    {
        "name": "order_status_lookup",
        "user_message": "What is the status of order 5001?",
        "expected_tool_calls": [
            {
                "name": "get_order",
                "arguments": {"order_id": 5001},
            }
        ],
        "expected_facts": ["Order 5001 is delivered"],
    },
    {
        "name": "product_availability_lookup",
        "user_message": "How many units of product 102 are currently in stock?",
        "expected_tool_calls": [
            {
                "name": "get_product",
                "arguments": {"product_id": 102},
            }
        ],
        "expected_facts": ["Product 102 has 2 units in stock"],
    },
    {
        "name": "missing_order_lookup",
        "user_message": "What is the status of order 9999?",
        "expected_tool_calls": [
            {
                "name": "get_order",
                "arguments": {"order_id": 9999},
            }
        ],
        "expected_facts": ["Order 9999 was not found"],
    },
    {
        "name": "multistep_order_and_product_lookup",
        "user_message": "What's the status of order 5001, and is its product currently in stock?",
        "expected_tool_calls": [
            {
                "name": "get_order",
                "arguments": {"order_id": 5001},
            },
            {
                "name": "get_product",
                "arguments": {"product_id": 102},
            },
        ],
        "expected_facts": [
            "Order 5001 is delivered, and product 102 has 2 units in stock."
        ],
    },
]
