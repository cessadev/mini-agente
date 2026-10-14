"""Módulo de envíos."""

TARIFAS = {"local": 5000, "nacional": 12000}


def calcular_total(peso_kg, zona):
    """Costo del envío: tarifa base de la zona más un recargo por kilo."""
    return TARIFAS[zona] + 800 * peso_kg
