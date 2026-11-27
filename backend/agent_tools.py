import os

from fastapi import HTTPException

from db import (
    find_product_db,
    find_order_db,
    prepare_order_db,
)

from schemas import (
    ProductLookupArgs,
    OrderLookupArgs,
    PrepareOrderRequest,
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
