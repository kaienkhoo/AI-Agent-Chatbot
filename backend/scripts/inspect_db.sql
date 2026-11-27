\echo '========== PRODUCTS =========='

SELECT id, name, stock
FROM products
ORDER BY id;

\echo '========== RECENT ORDERS =========='

SELECT *
FROM orders
ORDER BY id DESC
LIMIT 5;

\echo '========== RECENT DRAFTS =========='

SELECT id, product_id, quantity, status, order_id
FROM order_drafts
ORDER BY expires_at DESC
LIMIT 5;