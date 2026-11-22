from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, ConfigDict
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError
from fastapi.middleware.cors import CORSMiddleware
from db import find_product_db, find_order_db, get_all_products_db
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

AGENT_INSTRUCTIONS = """
You are a customer support assistant for an online store.

Use the available tools to look up products and orders.
For questions about current prices, inventory, or order status,
retrieve the relevant information rather than relying on
previous conversation messages.

Only state product and order facts supported by tool results.
Do not invent unavailable information.

If a record cannot be found, explain that clearly.
Keep your answers concise and helpful.
"""

TOOL_DEFINITIONS = [product_tool, order_tool]


class Chat(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    chats: list[Chat]


class ProductLookupArgs(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    product_id: int


class OrderLookupArgs(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    order_id: int


def find_product(product_id: int):
    return find_product_db(product_id)


def find_order(order_id: int):
    return find_order_db(order_id)


TOOL_HANDLERS = {
    "get_product": find_product,
    "get_order": find_order,
}

TOOL_ARG_MODELS = {
    "get_product": ProductLookupArgs,
    "get_order": OrderLookupArgs,
}


@app.post("/chat")
def chat(request: ChatHistory):
    try:
        response = client.responses.create(
            model="gpt-5.6-terra",
            instructions=AGENT_INSTRUCTIONS,
            input=[{"role": c.role, "content": c.content} for c in request.chats],
            tools=TOOL_DEFINITIONS,
            parallel_tool_calls=False,
        )

        max_tool_rounds = 3
        tool_rounds = 0

        while True:
            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not tool_calls:
                return {"answer": response.output_text}

            if tool_rounds >= max_tool_rounds:
                raise HTTPException(
                    status_code=500,
                    detail="Agent tool execution limit reached",
                )

            tool_call = tool_calls[0]

            tool_name = tool_call.name

            try:
                args = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid tool arguments received",
                )

            handler = TOOL_HANDLERS.get(tool_name)

            argument_model = TOOL_ARG_MODELS.get(tool_name)

            if handler is None or argument_model is None:
                raise HTTPException(
                    status_code=500,
                    detail="Unsupported tool requested",
                )

            try:
                validated = argument_model.model_validate(args)
            except ValidationError:
                raise HTTPException(
                    status_code=500,
                    detail="Invalid tool arguments received",
                )

            result = handler(**validated.model_dump())

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
                "call_id": tool_call.call_id,
                "output": json.dumps(result),
            }

            tool_rounds += 1

            response = client.responses.create(
                model="gpt-5.6-terra",
                instructions=AGENT_INSTRUCTIONS,
                input=[tool_result],
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                parallel_tool_calls=False,
                previous_response_id=response.id,
            )
    except APIConnectionError:
        raise HTTPException(
            status_code=503, detail="AI service temporarily unavailable"
        )


@app.get("/products")
def get_products():
    return get_all_products_db()


@app.get("/products/{product_id}")
def get_product(product_id: int):
    result = find_product(product_id)

    if result is not None:
        return result

    raise HTTPException(
        status_code=404, detail=f"Product with ID {product_id} not found"
    )
