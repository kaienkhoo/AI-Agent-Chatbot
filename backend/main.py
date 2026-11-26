from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, ConfigDict, Field
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from db import (
    find_product_db,
    find_order_db,
    get_all_products_db,
    prepare_order_db,
    confirm_order_db,
)
import json
import os

load_dotenv()

app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

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


prepare_order = {
    "type": "function",
    "name": "prepare_order",
    "description": (
        "Prepare a pending order draft when the customer explicitly "
        "requests to purchase a product. This does not place or confirm "
        "the order. Customer confirmation is required separately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "integer",
                "description": "The ID of the product to order",
            },
            "quantity": {
                "type": "integer",
                "description": "The quantity of the product to order",
            },
        },
        "required": ["product_id", "quantity"],
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

When the customer explicitly requests to purchase a product,
you may use prepare_order to create a pending draft.

Only prepare a draft when the product ID and quantity are clear.
If either is unclear, ask the customer for clarification.
Do not guess a product ID or quantity.

Preparing a draft does not place an order.
Tell the customer to review the draft and explicitly confirm
through the application's confirmation interface.

Never claim an order has been placed just because a draft exists.
"""

TOOL_DEFINITIONS = [
    product_tool,
    order_tool,
    prepare_order,
]


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


class PrepareOrderRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=10)


def find_product(product_id: int):
    return find_product_db(product_id)


def find_order(order_id: int):
    return find_order_db(order_id)


def prepare_order_tool(product_id: int, quantity: int):
    require_demo_ordering()

    try:
        draft = prepare_order_db(product_id, quantity)

        return draft
    except ValueError:
        return {"error": "product_unavailable_or_insufficient_stock"}


def get_available_tools():
    tools = [product_tool, order_tool]

    if os.getenv("ENABLE_DEMO_ORDERS") == "true":
        tools.append(prepare_order)

    return tools


TOOL_HANDLERS = {
    "get_product": find_product,
    "get_order": find_order,
    "prepare_order": prepare_order_tool,
}

TOOL_ARG_MODELS = {
    "get_product": ProductLookupArgs,
    "get_order": OrderLookupArgs,
    "prepare_order": PrepareOrderRequest,
}


def require_demo_ordering():
    if os.getenv("ENABLE_DEMO_ORDERS") != "true":
        raise HTTPException(status_code=404, detail="Not found")


@app.post("/chat")
def chat(request: ChatHistory):
    available_tools = get_available_tools()

    try:
        response = client.responses.create(
            model="gpt-5.6-terra",
            instructions=AGENT_INSTRUCTIONS,
            input=[{"role": c.role, "content": c.content} for c in request.chats],
            tools=available_tools,
            parallel_tool_calls=False,
        )

        max_tool_rounds = 3
        tool_rounds = 0

        created_draft = None

        while True:
            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not tool_calls:
                return {
                    "answer": response.output_text,
                    "order_draft": jsonable_encoder(created_draft),
                }

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

            if tool_name == "prepare_order" and created_draft is not None:
                result = {"error": "draft_already_prepared"}
            else:
                result = handler(**validated.model_dump())

                if (
                    tool_name == "prepare_order"
                    and isinstance(result, dict)
                    and "id" in result
                ):
                    created_draft = result

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
                "output": json.dumps(jsonable_encoder(result)),
            }

            tool_rounds += 1

            response = client.responses.create(
                model="gpt-5.6-terra",
                instructions=AGENT_INSTRUCTIONS,
                input=[tool_result],
                tools=available_tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                previous_response_id=response.id,
            )
    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable",
        )


@app.post("/order-drafts", status_code=201)
def prepare_order_route(request: PrepareOrderRequest):
    require_demo_ordering()

    try:
        prepare_order = prepare_order_db(request.product_id, request.quantity)

        return prepare_order
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Product unavailable or insufficient stock",
        )


@app.post("/order-drafts/{draft_id}/confirm")
def confirm_order_route(draft_id: str):
    require_demo_ordering()

    try:
        confirmed_order = confirm_order_db(draft_id)

        return confirmed_order
    except ValueError:
        raise HTTPException(status_code=400, detail="Order confirmation failed")


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
