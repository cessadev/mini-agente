STOCK_MINIMO = 5

class Inventario:
    def __init__(self):
        self.items = {}

    def agregar(self, nombre, cantidad):
        self.items[nombre] = self.items.get(nombre, 0) + cantidad

    def retirar(self, nombre, cantidad):
        if self.items.get(nombre, 0) < cantidad:
            raise ValueError("Stock insuficiente")
        self.items[nombre] -= cantidad
