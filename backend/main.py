from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from db import (
    get_all_products_db,
    prepare_order_db,
    confirm_order_db,
)

from schemas import ChatHistory, PrepareOrderRequest

from agent_tools import (
    find_product,
    require_demo_ordering,
)

from agent import run_agent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/chat")
def chat(request: ChatHistory):
    return run_agent(request)


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
