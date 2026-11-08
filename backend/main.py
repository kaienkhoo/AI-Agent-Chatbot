from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError
from fastapi.middleware.cors import CORSMiddleware

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


class Chat(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    chats: list[Chat]


@app.post("/chat")
def chat(request: ChatHistory):
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[{"role": c.role, "content": c.content} for c in request.chats],
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
    for product in products:
        if product["id"] == product_id:
            return product

    raise HTTPException(
        status_code=404, detail=f"Product with ID {product_id} not found"
    )
