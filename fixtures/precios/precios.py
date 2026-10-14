"""Cálculo del precio final que paga el cliente."""
from config import TASA_IVA, DESCUENTO_CLIENTE_FIEL


def precio_final(base, cliente_fiel=False):
    """Aplica el descuento (solo a clientes fieles) y después el IVA. Redondea al entero."""
    if cliente_fiel:
        base = base * (1 - DESCUENTO_CLIENTE_FIEL)
    return round(base * (1 + TASA_IVA))
