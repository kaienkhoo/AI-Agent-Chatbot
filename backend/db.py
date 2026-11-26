import os
import uuid
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from datetime import datetime, timezone

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


def create_order_db(product_id: int, quantity: int):
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero")

    with get_connection() as conn:
        updated_product = conn.execute(
            """
            UPDATE products
            SET stock = stock - %s
            WHERE id = %s
             AND stock >= %s
            RETURNING id, stock;
            """,
            (
                quantity,
                product_id,
                quantity,
            ),
        ).fetchone()

        if updated_product is None:
            raise ValueError("Product not found or insufficient stock")

        new_order = conn.execute(
            """
            INSERT INTO orders (product_id, quantity, status)
            VALUES (%s, %s, 'processing')
            RETURNING id, product_id, quantity, status;
            """,
            (
                product_id,
                quantity,
            ),
        ).fetchone()

    return new_order


def prepare_order_db(product_id: int, quantity: int):
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero")

    with get_connection() as conn:
        available_product = conn.execute(
            """
            SELECT id, name, price, stock
            FROM products
            WHERE id = %s
                AND stock >= %s;
            """,
            (
                product_id,
                quantity,
            ),
        ).fetchone()

        if available_product is None:
            raise ValueError("Product not found or insufficient stock")

        draft_id = str(uuid.uuid4())

        order_drafts = conn.execute(
            """
            INSERT INTO order_drafts (id, product_id, quantity)
            VALUES (%s, %s, %s)
            RETURNING id, product_id, quantity, status, expires_at;
            """,
            (
                draft_id,
                product_id,
                quantity,
            ),
        ).fetchone()

        total_price = available_product["price"] * quantity
        order_drafts["product_name"] = available_product["name"]
        order_drafts["unit_price"] = str(available_product["price"])
        order_drafts["total_price"] = str(total_price)

    return order_drafts


def confirm_order_db(draft_id: str):
    with get_connection() as conn:
        draft = conn.execute(
            """
            SELECT id, product_id, quantity, status, expires_at, order_id
            FROM order_drafts
            WHERE id = %s
            FOR UPDATE;
            """,
            (draft_id,),
        ).fetchone()

        if draft is None:
            raise ValueError("Draft not found")

        product_id = draft["product_id"]
        quantity = draft["quantity"]

        if draft["status"] == "confirmed":
            order_id = draft["order_id"]

            order = conn.execute(
                """
                SELECT *
                FROM orders
                WHERE id = %s;
                """,
                (order_id,),
            ).fetchone()

            return order

        if draft["expires_at"] <= datetime.now(timezone.utc):
            raise ValueError("Order draft has expired")

        updated_product = conn.execute(
            """
            UPDATE products
            SET stock = stock - %s
            WHERE id = %s
                AND stock >= %s
            RETURNING id, stock;
            """,
            (
                quantity,
                product_id,
                quantity,
            ),
        ).fetchone()

        if updated_product is None:
            raise ValueError("Product not found or insufficient stock")

        new_order = conn.execute(
            """
            INSERT INTO orders (product_id, quantity, status)
            VALUES (%s, %s, 'processing')
            RETURNING id, product_id, quantity, status;
            """,
            (
                product_id,
                quantity,
            ),
        ).fetchone()

        conn.execute(
            """
            UPDATE order_drafts
            SET status = 'confirmed',
                order_id = %s
            WHERE id = %s;
            """,
            (
                new_order["id"],
                draft_id,
            ),
        )

        return new_order
