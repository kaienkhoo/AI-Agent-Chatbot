import os

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
    )


def find_product_db(product_id: int):
    with get_connection() as conn:
        product = conn.execute(
            """
            SELECT id, name, price, stock
            FROM products
            WHERE id = %s
            """,
            (product_id,),
        ).fetchone()

    if product is None:
        return None

    product["price"] = str(product["price"])
    return product


def find_order_db(order_id: int):
    # Your implementation
    with get_connection() as conn:
        order = conn.execute(
            """
            SELECT id, product_id, quantity, status
            FROM orders
            WHERE id = %s;
            """,
            (order_id,),
        ).fetchone()

    if order is None:
        return None

    return order


def get_all_products_db():
    with get_connection() as conn:
        products = conn.execute(
            """
            SELECT id, name, price, stock
            FROM products
            ORDER BY id;
            """,
        ).fetchall()

    for product in products:
        product["price"] = str(product["price"])

    return products
