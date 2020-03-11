from generador_simulaciones_2 import _make_coordinate_structure
import numpy as np

x = np.array([
    [2, 3, -1, 4, 5, 100],
    [-2, 2, 4, 6, 5, 6]
])

y = np.array([
    [2, 3, -1, 4, 5, 300],
    [-2, 2, 4, 6, -5, 1]
])

print(_make_coordinate_structure(x, y, np.zeros(x.shape)))
