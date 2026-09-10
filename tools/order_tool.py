"""CrewAI Order Lookup Tool.

Looks up an order deterministically from the cached pandas dataset. Never
invents information: if the order does not exist it returns ``found: false``.
"""
from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from backend import data
from backend.schemas import OrderInfo


class OrderLookupInput(BaseModel):
    order_number: int = Field(..., description="The numeric order number to look up.")


def lookup_order(order_number: int) -> dict:
    """Plain Python entry point used by the deterministic pipeline."""
    info: OrderInfo = data.get_order(int(order_number))
    if not info.found:
        return {"found": False, "order_number": int(order_number)}
    return {
        "found": True,
        "order_number": info.order_number,
        "customer_name": info.customer_name,
        "status": info.status,
        "order_date": info.order_date,
        "total": info.total,
        "products": [
            {"product_id": p.product_id, "name": p.name} for p in info.products
        ],
    }


class OrderLookupTool(BaseTool):
    name: str = "order_lookup"
    description: str = (
        "Look up a customer order by its numeric order number. Returns the order "
        "status, date, total and products. Returns found=false if the order does "
        "not exist. Never invents order data."
    )
    args_schema: type[BaseModel] = OrderLookupInput

    def _run(self, order_number: int) -> str:
        try:
            return json.dumps(lookup_order(order_number), ensure_ascii=False)
        except Exception as exc:  # malformed input etc.
            return json.dumps(
                {"found": False, "order_number": order_number, "error": str(exc)},
                ensure_ascii=False,
            )