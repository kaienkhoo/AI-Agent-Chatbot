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


@app.post("/chat")
def chat(request: ChatHistory):
    try:
        response = client.responses.create(
            model="gpt-5.6-terra",
            input=[{"role": c.role, "content": c.content} for c in request.chats],
            tools=[product_tool],
            parallel_tool_calls=False,
        )

        for item in response.output:
            if item.type == "function_call":
                if item.name == "get_product":
                    args = json.loads(item.arguments)

                    product = find_product(args["product_id"])

                    if product is not None:
                        result = product
                    else:
                        result = {
                            "error": "product_not_found",
                            "product_id": args["product_id"],
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
