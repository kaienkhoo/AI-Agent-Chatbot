import json

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from openai import OpenAI, APIConnectionError
from dotenv import load_dotenv

from schemas import ChatHistory

from agent_tools import (
    get_available_tools,
    TOOL_HANDLERS,
    TOOL_ARG_MODELS,
)

load_dotenv()

client = OpenAI()

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


def run_agent(request: ChatHistory):
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
