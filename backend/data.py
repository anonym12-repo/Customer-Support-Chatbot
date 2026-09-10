"""Deterministic order / product data layer.

Loads the CSV files once, caches them, and builds an O(1) lookup index.
No eval() is ever used -- product id lists are parsed with ast.literal_eval.
"""
from __future__ import annotations

import ast
import logging
import threading
from pathlib import Path

import pandas as pd

from . import config
from .schemas import OrderInfo, ProductInfo

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_orders_by_id: dict[int, dict] | None = None
_products_by_id: dict[int, dict] | None = None


def _parse_product_ids(raw: object) -> list[int]:
    """Safely parse a column that holds a list-like string e.g. '[101,102]'."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    try:
        value = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError) as exc:
        logger.warning("Could not parse product ids %r: %s", raw, exc)
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return []


def _load(orders_csv: Path = config.ORDERS_CSV,
          products_csv: Path = config.PRODUCTS_CSV) -> None:
    global _orders_by_id, _products_by_id
    if _orders_by_id is not None and _products_by_id is not None:
        return
    with _lock:
        if _orders_by_id is not None and _products_by_id is not None:
            return
        try:
            orders_df = pd.read_csv(orders_csv)
            products_df = pd.read_csv(products_csv)
        except FileNotFoundError as exc:
            logger.error("Missing data file: %s", exc)
            _orders_by_id = {}
            _products_by_id = {}
            return
        except Exception as exc:  # corrupted CSV etc.
            logger.error("Failed to load order data: %s", exc)
            _orders_by_id = {}
            _products_by_id = {}
            return

        products: dict[int, dict] = {}
        for _, row in products_df.iterrows():
            pid = int(row["product_id"])
            products[pid] = {"product_id": pid, "name": str(row["name"])}

        orders: dict[int, dict] = {}
        for _, row in orders_df.iterrows():
            try:
                oid = int(row["order_id"])
            except (ValueError, KeyError):
                continue
            orders[oid] = {
                "order_id": oid,
                "customer_name": str(row.get("customer_name", "")),
                "order_date": str(row.get("order_date", "")),
                "total": float(row.get("total", 0.0)),
                "status": str(row.get("status", "")),
                "product_ids": _parse_product_ids(row.get("product_ids_ordered")),
            }
        _orders_by_id = orders
        _products_by_id = products
        logger.info("Loaded %d orders and %d products", len(orders), len(products))


def get_order(order_number: int) -> OrderInfo:
    """Deterministic O(1) order lookup. Never invents data."""
    _load()
    assert _orders_by_id is not None and _products_by_id is not None
    record = _orders_by_id.get(int(order_number))
    if record is None:
        return OrderInfo(found=False, order_number=int(order_number))
    products: list[ProductInfo] = []
    for pid in record["product_ids"]:
        prod = _products_by_id.get(pid)
        name = prod["name"] if prod else f"Unknown product {pid}"
        products.append(ProductInfo(product_id=pid, name=name))
    return OrderInfo(
        found=True,
        order_number=record["order_id"],
        customer_name=record["customer_name"],
        status=record["status"],
        order_date=record["order_date"],
        total=record["total"],
        products=products,
    )


def order_exists(order_number: int) -> bool:
    _load()
    assert _orders_by_id is not None
    return int(order_number) in _orders_by_id


def all_order_ids() -> list[int]:
    _load()
    assert _orders_by_id is not None
    return sorted(_orders_by_id.keys())


def reset_cache() -> None:
    """Force a reload (used by tests)."""
    global _orders_by_id, _products_by_id
    with _lock:
        _orders_by_id = None
        _products_by_id = None