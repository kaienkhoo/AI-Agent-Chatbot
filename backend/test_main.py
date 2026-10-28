from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from openai import APIConnectionError
from main import app, client

test_client = TestClient(app)


def test_chat_returns_503_when_openai_unreachable():
    fake_error = APIConnectionError(request=Mock())

    with patch.object(client.responses, "create", side_effect=fake_error):
        response = test_client.post("/chat", json={"message": "Hello!"})

    # Your assertions go here
    assert response.status_code == 503
    assert response.json() == {"detail": "AI service temporarily unavailable"}
