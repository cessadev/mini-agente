"""Módulo de ventas."""


def calcular_total(items):
    """Suma precio * cantidad de cada ítem del pedido."""
    return sum(i["precio"] * i["cantidad"] for i in items)
