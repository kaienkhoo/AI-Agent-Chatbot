import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from openai import APIConnectionError
from main import (
    app,
    find_product,
)

from agent_tools import find_order
from agent import client, AGENT_INSTRUCTIONS, TOOL_HANDLERS

test_client = TestClient(app)


def test_chat_returns_503_when_openai_unreachable():
    fake_error = APIConnectionError(request=Mock())

    with patch.object(client.responses, "create", side_effect=fake_error):
        response = test_client.post(
            "/chat", json={"chats": [{"role": "user", "content": "Hello!"}]}
        )

    # Your assertions go here
    assert response.status_code == 503
    assert response.json() == {"detail": "AI service temporarily unavailable"}


def test_find_product_returns_correct_name():
    product_id = 101

    result = find_product(product_id)

    assert result is not None
    assert result["name"] == "Wireless Mouse"


def test_find_product_returns_none_for_missing_product():
    # Arrange
    product_id = 999

    # Act
    result = find_product(product_id)

    # Assert
    assert result is None


def test_get_product_returns_correct_product():
    with patch("agent_tools.find_product_db") as mock_lookup:
        mock_lookup.return_value = {
            "id": 101,
            "name": "Wireless Mouse",
            "price": "49.90",
            "stock": 5,
        }

        response = test_client.get("/products/101")

        assert response.status_code == 200
        assert response.json() == {
            "id": 101,
            "name": "Wireless Mouse",
            "price": "49.90",
            "stock": 5,
        }
        mock_lookup.assert_called_once_with(101)


def test_get_product_returns_404_for_missing_product():
    response = test_client.get("/products/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Product with ID 999 not found"}


def test_chat_returns_answer_without_tool_call():
    # Arrange
    fake_ai_response = Mock(output=[], output_text="Hello from fake OpenAI!")

    with patch.object(
        client.responses, "create", return_value=fake_ai_response
    ) as mock_create:
        response = test_client.post(
            "/chat", json={"chats": [{"role": "user", "content": "Hello!"}]}
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Hello from fake OpenAI!",
        "order_draft": None,
    }
    assert mock_create.call_count == 1


def test_chat_returns_answer_with_tool_call():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id":102}',
        call_id="call_123",
    )

    fake_tool_call.name = "get_product"

    fake_first_response = Mock(
        id="response_123",
        output=[fake_tool_call],
    )

    fake_second_response = Mock(
        output=[],
        output_text="Product 102 has 2 units available",
    )

    fake_api = Mock()

    fake_api.side_effect = [fake_first_response, fake_second_response]

    with patch.object(client.responses, "create", fake_api) as mock_create:
        response = test_client.post(
            "/chat",
            json={"chats": [{"role": "user", "content": "Product 102 stocks"}]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Product 102 has 2 units available",
        "order_draft": None,
    }
    assert mock_create.call_count == 2

    second_call = mock_create.call_args_list[1]  # Second call
    tool_message = second_call.kwargs["input"][0]
    product_data = json.loads(tool_message["output"])

    assert tool_message["call_id"] == "call_123"
    assert product_data["id"] == 102
    assert product_data["stock"] == 2
    assert second_call.kwargs["previous_response_id"] == "response_123"


def test_chat_handles_missing_product():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id":999}',
        call_id="call_123",
    )

    fake_tool_call.name = "get_product"

    fake_first_response = Mock(
        id="response_123",
        output=[fake_tool_call],
    )

    fake_second_response = Mock(
        output=[],
        output_text="Sorry, product 999 was not found.",
    )

    fake_api = Mock()

    fake_api.side_effect = [fake_first_response, fake_second_response]

    with patch.object(client.responses, "create", fake_api) as mock_create:
        response = test_client.post(
            "/chat",
            json={"chats": [{"role": "user", "content": "Product 999 stocks"}]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Sorry, product 999 was not found.",
        "order_draft": None,
    }
    assert mock_create.call_count == 2

    second_call = mock_create.call_args_list[1]
    tool_message = second_call.kwargs["input"][0]
    product_data = json.loads(tool_message["output"])

    assert tool_message["call_id"] == "call_123"
    assert product_data["error"] == "product_not_found"
    assert product_data["product_id"] == 999


def test_find_order_returns_correct_order():
    order_id = 5001

    result = find_order(order_id)

    assert result is not None
    assert result["product_id"] == 102


def test_find_order_returns_none_for_missing_order():
    order_id = 9999

    result = find_order(order_id)

    assert result is None


def test_find_order_delegates_to_database():
    expected_order = {
        "id": 5001,
        "product_id": 102,
        "quantity": 1,
        "status": "delivered",
    }

    with patch("agent_tools.find_order_db", return_value=expected_order) as mock_lookup:
        result = find_order(5001)

        assert result == expected_order
        mock_lookup.assert_called_once_with(5001)


def test_chat_returns_answer_with_order_tool():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"order_id":5001}',
        call_id="call_123",
    )

    fake_tool_call.name = "get_order"

    fake_first_response = Mock(
        id="response_123",
        output=[fake_tool_call],
    )

    fake_second_response = Mock(
        output=[],
        output_text="Order 5001 has been delivered.",
    )

    fake_api = Mock()

    fake_api.side_effect = [fake_first_response, fake_second_response]

    with patch.object(client.responses, "create", fake_api) as mock_create:
        response = test_client.post(
            "/chat",
            json={"chats": [{"role": "user", "content": "Order 5001"}]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Order 5001 has been delivered.",
        "order_draft": None,
    }
    assert mock_create.call_count == 2

    second_call = mock_create.call_args_list[1]  # Second call
    tool_message = second_call.kwargs["input"][0]
    order_data = json.loads(tool_message["output"])

    assert tool_message["call_id"] == "call_123"
    assert order_data["product_id"] == 102
    assert order_data["id"] == 5001
    assert order_data["status"] == "delivered"
    assert second_call.kwargs["previous_response_id"] == "response_123"


def test_chat_handles_missing_order():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"order_id":9999}',
        call_id="call_123",
    )

    fake_tool_call.name = "get_order"

    fake_first_response = Mock(
        id="response_123",
        output=[fake_tool_call],
    )

    fake_second_response = Mock(
        output=[],
        output_text="Sorry, order 9999 was not found.",
    )

    fake_api = Mock()

    fake_api.side_effect = [fake_first_response, fake_second_response]

    with patch.object(client.responses, "create", fake_api) as mock_create:
        response = test_client.post(
            "/chat",
            json={"chats": [{"role": "user", "content": "Order 9999"}]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Sorry, order 9999 was not found.",
        "order_draft": None,
    }
    assert mock_create.call_count == 2

    second_call = mock_create.call_args_list[1]
    tool_message = second_call.kwargs["input"][0]
    order_data = json.loads(tool_message["output"])

    assert second_call.kwargs["previous_response_id"] == "response_123"
    assert tool_message["call_id"] == "call_123"
    assert order_data["error"] == "order_not_found"
    assert order_data["order_id"] == 9999


def test_chat_multi_step_order_then_product():
    fake_order_call = Mock(
        type="function_call",
        arguments='{"order_id":5001}',
        call_id="call_order_1",
    )

    fake_order_call.name = "get_order"

    fake_product_call = Mock(
        type="function_call",
        arguments='{"product_id":102}',
        call_id="call_product_1",
    )

    fake_product_call.name = "get_product"

    fake_response_1 = Mock(
        id="response_1",
        output=[fake_order_call],
    )

    fake_response_2 = Mock(
        id="response_2",
        output=[fake_product_call],
    )

    fake_response_3 = Mock(
        output=[],
        output_text="Order 5001 status is delivered and stock is 2.",
    )

    with patch.object(client.responses, "create") as fake_api:
        fake_api.side_effect = [
            fake_response_1,
            fake_response_2,
            fake_response_3,
        ]

        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {
                        "role": "user",
                        "content": "What's the status of order 5001 and is its product in stock?",
                    }
                ]
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "Order 5001 status is delivered and stock is 2.",
            "order_draft": None,
        }
        assert fake_api.call_count == 3

        second_api_call = fake_api.call_args_list[1]

        order_tool_message = second_api_call.kwargs["input"][0]

        order_data = json.loads(order_tool_message["output"])

        assert order_data["id"] == 5001
        assert order_data["product_id"] == 102
        assert order_data["status"] == "delivered"

        assert order_tool_message["call_id"] == "call_order_1"

        assert second_api_call.kwargs["previous_response_id"] == "response_1"

        third_api_call = fake_api.call_args_list[2]

        product_tool_message = third_api_call.kwargs["input"][0]

        product_data = json.loads(product_tool_message["output"])

        assert product_data["id"] == 102
        assert product_data["stock"] == 2
        assert product_tool_message["call_id"] == "call_product_1"
        assert third_api_call.kwargs["previous_response_id"] == "response_2"

        for api_call in fake_api.call_args_list:
            assert api_call.kwargs["instructions"] == AGENT_INSTRUCTIONS


def make_product_response(number):
    tool_call = Mock(
        type="function_call",
        arguments='{"product_id":102}',
        call_id=f"call_product_{number}",
    )

    tool_call.name = "get_product"

    return Mock(
        id=f"response_{number}",
        output=[tool_call],
    )


def test_chat_stops_when_tool_limit_reached():
    product_spy = Mock(wraps=find_product)
    fake_responses = [make_product_response(i) for i in range(1, 5)]

    with patch.dict(TOOL_HANDLERS, {"get_product": product_spy}):
        with patch.object(client.responses, "create") as fake_api:
            fake_api.side_effect = fake_responses

            response = test_client.post(
                "/chat",
                json={
                    "chats": [
                        {
                            "role": "user",
                            "content": "What's the status of order 5001 and is its product in stock?",
                        }
                    ]
                },
            )

            assert response.status_code == 500
            assert response.json() == {"detail": "Agent tool execution limit reached"}
            assert fake_api.call_count == 4
            assert product_spy.call_count == 3


def test_chat_rejects_malformed_tool_json():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id":102',
        call_id="call_invalid_1",
    )

    fake_tool_call.name = "get_product"

    fake_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    with patch.object(client.responses, "create") as fake_api:
        fake_api.return_value = fake_response

        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {
                        "role": "user",
                        "content": "Is product 102 in stock?",
                    }
                ]
            },
        )

        assert response.status_code == 500
        assert response.json() == {"detail": "Invalid tool arguments received"}
        assert fake_api.call_count == 1


def test_chat_rejects_invalid_product_id_type():
    product_spy = Mock(wraps=find_product)

    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id": "102"}',
        call_id="call_invalid_type",
    )

    fake_tool_call.name = "get_product"

    fake_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    with patch.dict(
        TOOL_HANDLERS,
        {"get_product": product_spy},
    ):
        with patch.object(client.responses, "create") as fake_api:
            fake_api.return_value = fake_response

            response = test_client.post(
                "/chat",
                json={
                    "chats": [
                        {
                            "role": "user",
                            "content": "Is product 102 in stock?",
                        }
                    ]
                },
            )

            assert response.status_code == 500
            assert response.json() == {"detail": "Invalid tool arguments received"}
            assert fake_api.call_count == 1
            assert product_spy.call_count == 0


def test_chat_rejects_unknown_tool():
    fake_tool_call = Mock(
        type="function_call",
        arguments="{}",
        call_id="call_unknown_1",
    )

    fake_tool_call.name = "delete_all_orders"

    fake_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    with patch.object(client.responses, "create") as fake_api:
        fake_api.return_value = fake_response

        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {
                        "role": "user",
                        "content": "Please delete all orders",
                    }
                ],
            },
        )

        assert response.status_code == 500
        assert response.json() == {"detail": "Unsupported tool requested"}
        assert fake_api.call_count == 1


def test_chat_returns_created_order_draft():
    fake_draft = {
        "id": "test-draft-123",
        "product_id": 101,
        "quantity": 1,
        "status": "pending",
        "product_name": "Wireless Mouse",
        "unit_price": "49.90",
        "total_price": "49.90",
    }

    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id": 101, "quantity": 1}',
        call_id="call_prepare_1",
    )

    fake_tool_call.name = "prepare_order"

    fake_first_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    fake_second_response = Mock(
        output=[],
        output_text="Your order draft is ready.",
    )

    with (
        patch.dict("os.environ", {"ENABLE_DEMO_ORDERS": "true"}),
        patch.object(
            client.responses,
            "create",
            side_effect=[fake_first_response, fake_second_response],
        ) as mock_create,
        patch(
            "agent_tools.prepare_order_db",
            return_value=fake_draft,
        ) as mock_prepare,
    ):
        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {"role": "user", "content": "Prepare an order for product 101."}
                ]
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "Your order draft is ready.",
            "order_draft": fake_draft,
        }
        mock_prepare.assert_called_once_with(101, 1)
        assert mock_create.call_count == 2


def test_chat_handles_failed_order_preparation():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id": 101, "quantity": 10}',
        call_id="call_prepare_1",
    )

    fake_tool_call.name = "prepare_order"

    fake_first_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    fake_second_response = Mock(
        output=[],
        output_text="Sorry, insufficient stock.",
    )

    with (
        patch.dict("os.environ", {"ENABLE_DEMO_ORDERS": "true"}),
        patch.object(
            client.responses,
            "create",
            side_effect=[fake_first_response, fake_second_response],
        ) as mock_create,
        patch(
            "agent_tools.prepare_order_db",
            side_effect=ValueError("Insufficient stock"),
        ) as mock_prepare,
    ):
        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {"role": "user", "content": "Prepare an order for product 101."}
                ]
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "Sorry, insufficient stock.",
            "order_draft": None,
        }
        mock_prepare.assert_called_once_with(101, 10)
        assert mock_create.call_count == 2
        tool_output = mock_create.call_args_list[1].kwargs["input"][0]["output"]
        assert json.loads(tool_output) == {
            "error": "product_unavailable_or_insufficient_stock"
        }


def test_chat_prevents_duplicate_order_drafts():
    fake_draft = {
        "id": "test-draft-123",
        "product_id": 101,
        "quantity": 1,
        "status": "pending",
        "product_name": "Wireless Mouse",
        "unit_price": "49.90",
        "total_price": "49.90",
    }

    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id": 101, "quantity": 1}',
        call_id="call_prepare_1",
    )

    fake_tool_call.name = "prepare_order"

    fake_first_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    fake_second_tool_call = Mock(
        type="function_call",
        arguments='{"product_id": 101, "quantity": 1}',
        call_id="call_prepare_2",
    )

    fake_second_tool_call.name = "prepare_order"

    fake_second_response = Mock(
        id="response_2",
        output=[fake_second_tool_call],
    )

    fake_third_response = Mock(
        output=[],
        output_text="Your order draft is ready.",
    )

    with (
        patch.dict("os.environ", {"ENABLE_DEMO_ORDERS": "true"}),
        patch.object(
            client.responses,
            "create",
            side_effect=[
                fake_first_response,
                fake_second_response,
                fake_third_response,
            ],
        ) as mock_create,
        patch(
            "agent_tools.prepare_order_db",
            return_value=fake_draft,
        ) as mock_prepare,
    ):
        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {"role": "user", "content": "Prepare an order for product 101."}
                ]
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "Your order draft is ready.",
            "order_draft": fake_draft,
        }
        mock_prepare.assert_called_once_with(101, 1)
        assert mock_create.call_count == 3
        tool_output = mock_create.call_args_list[2].kwargs["input"][0]["output"]
        assert json.loads(tool_output) == {"error": "draft_already_prepared"}


def test_chat_blocks_ordering_when_disabled():
    fake_tool_call = Mock(
        type="function_call",
        arguments='{"product_id": 101, "quantity": 1}',
        call_id="call_prepare_1",
    )

    fake_tool_call.name = "prepare_order"

    fake_response = Mock(
        id="response_1",
        output=[fake_tool_call],
    )

    with (
        patch.dict("os.environ", {"ENABLE_DEMO_ORDERS": "false"}),
        patch.object(
            client.responses,
            "create",
            return_value=fake_response,
        ) as mock_create,
        patch("agent_tools.prepare_order_db") as mock_prepare,
    ):
        response = test_client.post(
            "/chat",
            json={
                "chats": [
                    {"role": "user", "content": "Prepare an order for product 101."}
                ]
            },
        )

        assert response.status_code == 404
        assert response.json() == {
            "detail": "Not found",
        }
        assert mock_create.call_count == 1
        mock_prepare.assert_not_called()
        first_call = mock_create.call_args_list[0]
        advertised_tools = first_call.kwargs["tools"]
        tool_names = [tool["name"] for tool in advertised_tools]
        assert "prepare_order" not in tool_names
