# Example LP Problem:
# Maximize z = 3x1 + 2x2
# Subject to:
# 2x1 + x2 ≤ 4
# x1 + 2x2 ≤ 3
# x1, x2 ≥ 0
import numpy as np
from revisedSimplex import RevisedSimplex

A = np.array([
    [2, 1, 1, 0],  #Coefficients of x1, x2, s1, s2
    [1, 2, 0, 1]
], dtype=float)

b = np.array([4, 3], dtype=float)
c = np.array([3, 2, 0, 0], dtype=float)
initial_basis = [2, 3]  #Initial slack variables s1, s2

solver = RevisedSimplex(A, b, c, initial_basis)
x, obj = solver.solve()

print("Optimal Solution:")
print("x =", x[:2])  #x1 and x2
print("Objective Value:", obj)