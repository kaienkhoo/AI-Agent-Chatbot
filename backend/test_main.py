import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from openai import APIConnectionError
from main import app, client, orders, find_product, find_order
from copy import deepcopy

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
    response = test_client.get("/products/101")

    # Your assertions here
    assert response.status_code == 200
    assert response.json() == {
        "id": 101,
        "name": "Wireless Mouse",
        "price": 49.90,
        "stock": 5,
    }


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
    assert response.json() == {"answer": "Hello from fake OpenAI!"}
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
    assert response.json() == {"answer": "Product 102 has 2 units available"}
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
    assert response.json() == {"answer": "Sorry, product 999 was not found."}
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


def test_find_order_does_not_modify_orders():
    original_order = deepcopy(orders)

    find_order(5001)

    assert orders == original_order


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
    assert response.json() == {"answer": "Order 5001 has been delivered."}
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
    assert response.json() == {"answer": "Sorry, order 9999 was not found."}
    assert mock_create.call_count == 2

    second_call = mock_create.call_args_list[1]
    tool_message = second_call.kwargs["input"][0]
    order_data = json.loads(tool_message["output"])

    assert second_call.kwargs["previous_response_id"] == "response_123"
    assert tool_message["call_id"] == "call_123"
    assert order_data["error"] == "order_not_found"
    assert order_data["order_id"] == 9999
