import numpy as np

class RevisedSimplex:
    def __init__(self, A, b, c, basis, max_iter=1000, tol=1e-9):
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.c = c.astype(float)
        self.basis = basis
        self.m, self.n = self.A.shape
        self.max_iter = max_iter
        self.tol = tol
        self.x = np.zeros(self.n)
        self.obj_val = 0.0
        self.iterations = 0

        self._validate_basis()

    def _validate_basis(self):
        if len(self.basis) != self.m:
            raise ValueError("Basis must contain exactly m indices.")

        self.B = self.A[:, self.basis]

        try:
            self.B_inv = np.linalg.inv(self.B)
        except np.linalg.LinAlgError:
            raise ValueError("Initial basis matrix is singular.")

        self.x_b = self.B_inv @ self.b  

        if not np.all(self.x_b >= -self.tol):
            raise ValueError("Initial basis is not feasible.")

        self.x[self.basis] = self.x_b
        self.obj_val = self.c @ self.x

    def solve(self):
        for _ in range(self.max_iter):

            y = self.c[self.basis] @ self.B_inv
            non_basis = [j for j in range(self.n) if j not in self.basis]
            reduced_costs = [self.c[j] - y @ self.A[:, j] for j in non_basis]

            if all(rc <= self.tol for rc in reduced_costs):
                print(f"Optimal solution found in {self.iterations} iterations.")
                self.obj_val = self.c @ self.x
                return self.x, self.obj_val

            entering_idx = np.argmax(reduced_costs)
            entering_var = non_basis[entering_idx]

            d = self.B_inv @ self.A[:, entering_var]

            if np.all(d <= self.tol):
                print("Problem is unbounded.")
                return None, np.inf

            ratios = np.full(self.m, np.inf)
            for i in range(self.m):
                if d[i] > self.tol:
                    ratios[i] = self.x_b[i] / d[i]
            leaving_pos = np.argmin(ratios)
            leaving_var = self.basis[leaving_pos]

            self.basis[leaving_pos] = entering_var
            self.B = self.A[:, self.basis]
            try:
                self.B_inv = np.linalg.inv(self.B)
            except np.linalg.LinAlgError:
                raise ValueError("Basis matrix became singular during update.")

            self.x_b = self.B_inv @ self.b
            if not np.all(self.x_b >= -self.tol):
                raise ValueError("Basis update resulted in infeasible solution.")
            self.x.fill(0)
            self.x[self.basis] = self.x_b
            self.obj_val = self.c @ self.x
            self.iterations += 1

        print("Maximum iterations reached without convergence.")
        return self.x, self.obj_val
