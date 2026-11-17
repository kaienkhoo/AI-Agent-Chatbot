from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError
from fastapi.middleware.cors import CORSMiddleware
import json

load_dotenv()

app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

products = [
    {"id": 101, "name": "Wireless Mouse", "price": 49.90, "stock": 5},
    {"id": 102, "name": "Wireless Headphone", "price": 99.90, "stock": 2},
    {"id": 103, "name": "Wireless Keyboard", "price": 149.90, "stock": 3},
]

orders = [
    {
        "id": 5001,
        "product_id": 102,
        "quantity": 1,
        "status": "delivered",
    },
    {
        "id": 5002,
        "product_id": 101,
        "quantity": 2,
        "status": "processing",
    },
]

product_tool = {
    "type": "function",
    "name": "get_product",
    "description": "Retrieve product details using a product ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "integer",
                "description": "The ID of the product to retrieve",
            }
        },
        "required": ["product_id"],
        "additionalProperties": False,
    },
    "strict": True,
}

order_tool = {
    "type": "function",
    "name": "get_order",
    "description": "Retrieve order details using order ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "integer",
                "description": "The ID of the order to retrieve",
            }
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    "strict": True,
}


TOOL_DEFINITIONS = [product_tool, order_tool]


class Chat(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    chats: list[Chat]


def find_product(product_id: int):
    for product in products:
        if product["id"] == product_id:
            return product

    return None


def find_order(order_id: int):
    for order in orders:
        if order["id"] == order_id:
            return order

    return None


TOOL_HANDLERS = {
    "get_product": find_product,
    "get_order": find_order,
}


@app.post("/chat")
def chat(request: ChatHistory):
    try:
        response = client.responses.create(
            model="gpt-5.6-terra",
            input=[{"role": c.role, "content": c.content} for c in request.chats],
            tools=TOOL_DEFINITIONS,
            parallel_tool_calls=False,
        )

        for item in response.output:
            if item.type == "function_call":
                tool_name = item.name

                args = json.loads(item.arguments)

                handler = TOOL_HANDLERS.get(tool_name)

                if handler is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Unsupported tool requested",
                    )

                result = handler(**args)

                if result is None:
                    if tool_name == "get_product":
                        result = {
                            "error": "product_not_found",
                            "product_id": args["product_id"],
                        }
                    elif tool_name == "get_order":
                        result = {
                            "error": "order_not_found",
                            "order_id": args["order_id"],
                        }

                tool_result = {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }

                response = client.responses.create(
                    model="gpt-5.6-terra",
                    input=[tool_result],
                    tool_choice="none",
                    previous_response_id=response.id,
                )

        return {"answer": response.output_text}
    except APIConnectionError:
        raise HTTPException(
            status_code=503, detail="AI service temporarily unavailable"
        )


@app.get("/products")
def get_products():
    return {"product_list": products}


@app.get("/products/{product_id}")
def get_product(product_id: int):
    result = find_product(product_id)

    if result is not None:
        return result

    raise HTTPException(
        status_code=404, detail=f"Product with ID {product_id} not found"
    )
