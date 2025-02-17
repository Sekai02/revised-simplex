import numpy as np
import pulp
from revisedSimplex import RevisedSimplex

def run_test(test_name, test_func):
    try:
        test_func()
        print(f"✅ {test_name} passed.")
    except AssertionError as e:
        print(f"❌ {test_name} failed: {e}")
    except Exception as e:
        print(f"❌ {test_name} failed with an unexpected error: {e}")


def extract_original_vars(x, num_original):
    """Assumes the first 'num_original' entries in x are the original decision variables."""
    return x[:num_original]


def test_lp1():
    """
    LP1:
    Maximize: 3x1 + 2x2
    Subject to:
      x1 + x2 <= 4
      2x1 + x2 <= 5
    Standard form (with slacks s1 and s2):
      x1 + x2 + s1 = 4
      2x1 + x2 + s2 = 5
    Basis: [s1, s2] -> indices [2, 3]
    """
    A = np.array([[1, 1, 1, 0],
                  [2, 1, 0, 1]], dtype=float)
    b = np.array([4, 5], dtype=float)
    c = np.array([3, 2, 0, 0], dtype=float)
    basis = [2, 3]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()

    # Solve using PuLP
    prob = pulp.LpProblem("LP1", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    prob += 3 * x1 + 2 * x2, "Objective"
    prob += x1 + x2 <= 4, "Constraint1"
    prob += 2 * x1 + x2 <= 5, "Constraint2"
    prob.solve()

    x1_pulp = pulp.value(x1)
    x2_pulp = pulp.value(x2)
    obj_pulp = pulp.value(prob.objective)

    # Compare original decision variables (first two variables)
    orig_rs = extract_original_vars(x_rs, 2)
    assert np.isclose(orig_rs[0], x1_pulp, atol=1e-6), f"x1: {orig_rs[0]} vs {x1_pulp}"
    assert np.isclose(orig_rs[1], x2_pulp, atol=1e-6), f"x2: {orig_rs[1]} vs {x2_pulp}"
    assert np.isclose(obj_rs, obj_pulp, atol=1e-6), f"obj: {obj_rs} vs {obj_pulp}"


def test_lp2():
    """
    LP2:
    Maximize: x1 + 2x2
    Subject to:
      x1 + x2 <= 3
      x1 + 2x2 <= 4
    Standard form (with slacks s1 and s2):
      x1 + x2 + s1 = 3
      x1 + 2x2 + s2 = 4
    Basis: [s1, s2] -> indices [2, 3]
    """
    A = np.array([[1, 1, 1, 0],
                  [1, 2, 0, 1]], dtype=float)
    b = np.array([3, 4], dtype=float)
    c = np.array([1, 2, 0, 0], dtype=float)
    basis = [2, 3]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()

    # Solve with PuLP
    prob = pulp.LpProblem("LP2", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    prob += x1 + 2 * x2, "Objective"
    prob += x1 + x2 <= 3, "Constraint1"
    prob += x1 + 2 * x2 <= 4, "Constraint2"
    prob.solve()

    x1_pulp = pulp.value(x1)
    x2_pulp = pulp.value(x2)
    obj_pulp = pulp.value(prob.objective)

    orig_rs = extract_original_vars(x_rs, 2)
    assert np.isclose(orig_rs[0], x1_pulp, atol=1e-6), f"x1: {orig_rs[0]} vs {x1_pulp}"
    assert np.isclose(orig_rs[1], x2_pulp, atol=1e-6), f"x2: {orig_rs[1]} vs {x2_pulp}"
    assert np.isclose(obj_rs, obj_pulp, atol=1e-6), f"obj: {obj_rs} vs {obj_pulp}"


def test_lp3():
    """
    LP3:
    Maximize: 5x1 + 4x2
    Subject to:
      x1 + x2 <= 5
      x1 <= 3
      x2 <= 3
    Standard form (with slacks s1, s2, s3):
      x1 + x2 + s1 = 5
      x1      + s2 = 3
           x2 + s3 = 3
    Basis: [s1, s2, s3] -> indices [2, 3, 4]
    """
    A = np.array([[1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1]], dtype=float)
    b = np.array([5, 3, 3], dtype=float)
    c = np.array([5, 4, 0, 0, 0], dtype=float)
    basis = [2, 3, 4]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()

    # PuLP formulation
    prob = pulp.LpProblem("LP3", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    prob += 5 * x1 + 4 * x2, "Objective"
    prob += x1 + x2 <= 5, "Constraint1"
    prob += x1 <= 3, "Constraint2"
    prob += x2 <= 3, "Constraint3"
    prob.solve()

    x1_pulp = pulp.value(x1)
    x2_pulp = pulp.value(x2)
    obj_pulp = pulp.value(prob.objective)

    orig_rs = extract_original_vars(x_rs, 2)
    assert np.isclose(orig_rs[0], x1_pulp, atol=1e-6), f"x1: {orig_rs[0]} vs {x1_pulp}"
    assert np.isclose(orig_rs[1], x2_pulp, atol=1e-6), f"x2: {orig_rs[1]} vs {x2_pulp}"
    assert np.isclose(obj_rs, obj_pulp, atol=1e-6), f"obj: {obj_rs} vs {obj_pulp}"


def test_lp4():
    """
    LP4 (Degenerate):
    Maximize: x1 + x2
    Subject to:
      x1 + x2 <= 1
      x1 <= 0.5
      x2 <= 0.5
    Standard form (with slacks s1, s2, s3):
      x1 + x2 + s1 = 1
      x1      + s2 = 0.5
           x2 + s3 = 0.5
    Basis: [s1, s2, s3] -> indices [2, 3, 4]
    """
    A = np.array([[1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1]], dtype=float)
    b = np.array([1, 0.5, 0.5], dtype=float)
    c = np.array([1, 1, 0, 0, 0], dtype=float)
    basis = [2, 3, 4]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()

    # PuLP formulation
    prob = pulp.LpProblem("LP4", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    prob += x1 + x2, "Objective"
    prob += x1 + x2 <= 1, "Constraint1"
    prob += x1 <= 0.5, "Constraint2"
    prob += x2 <= 0.5, "Constraint3"
    prob.solve()

    x1_pulp = pulp.value(x1)
    x2_pulp = pulp.value(x2)
    obj_pulp = pulp.value(prob.objective)

    orig_rs = extract_original_vars(x_rs, 2)
    assert np.isclose(orig_rs[0], x1_pulp, atol=1e-6), f"x1: {orig_rs[0]} vs {x1_pulp}"
    assert np.isclose(orig_rs[1], x2_pulp, atol=1e-6), f"x2: {orig_rs[1]} vs {x2_pulp}"
    assert np.isclose(obj_rs, obj_pulp, atol=1e-6), f"obj: {obj_rs} vs {obj_pulp}"


def test_lp5():
    """
    LP5:
    Maximize: 2x1 + 3x2 + x3
    Subject to:
      x1 + x2 + x3 <= 10
      2x1 + x2 <= 8
      x2 + 3x3 <= 12
    Standard form (with slacks s1, s2, s3):
      x1 + x2 + x3 + s1 = 10
      2x1 + x2      + s2 = 8
           x2 + 3x3 + s3 = 12
    Basis: [s1, s2, s3] -> indices [3, 4, 5]
    """
    A = np.array([[1, 1, 1, 1, 0, 0],
                  [2, 1, 0, 0, 1, 0],
                  [0, 1, 3, 0, 0, 1]], dtype=float)
    b = np.array([10, 8, 12], dtype=float)
    c = np.array([2, 3, 1, 0, 0, 0], dtype=float)
    basis = [3, 4, 5]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()

    # PuLP formulation
    prob = pulp.LpProblem("LP5", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    x3 = pulp.LpVariable("x3", lowBound=0)
    prob += 2 * x1 + 3 * x2 + x3, "Objective"
    prob += x1 + x2 + x3 <= 10, "Constraint1"
    prob += 2 * x1 + x2 <= 8, "Constraint2"
    prob += x2 + 3 * x3 <= 12, "Constraint3"
    prob.solve()

    orig_rs = extract_original_vars(x_rs, 3)
    x1_pulp = pulp.value(x1)
    x2_pulp = pulp.value(x2)
    x3_pulp = pulp.value(x3)
    obj_pulp = pulp.value(prob.objective)

    assert np.isclose(orig_rs[0], x1_pulp, atol=1e-6), f"x1: {orig_rs[0]} vs {x1_pulp}"
    assert np.isclose(orig_rs[1], x2_pulp, atol=1e-6), f"x2: {orig_rs[1]} vs {x2_pulp}"
    assert np.isclose(orig_rs[2], x3_pulp, atol=1e-6), f"x3: {orig_rs[2]} vs {x3_pulp}"
    assert np.isclose(obj_rs, obj_pulp, atol=1e-6), f"obj: {obj_rs} vs {obj_pulp}"


def test_unbounded():
    """
    LP Unbounded:
    Maximize: x1 + x2
    Subject to (equality constraint):
      x1 - x2 + s1 = 0
    With nonnegativity constraints.
    The feasible region is { x1,x2 >= 0 and x1 = x2 }, so the objective 2x1 is unbounded.
    Basis: [s1] -> index [2]
    """
    A = np.array([[1, -1, 1]], dtype=float)
    b = np.array([0], dtype=float)
    c = np.array([1, 1, 0], dtype=float)
    basis = [2]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()
    # Our solver returns (None, np.inf) for unbounded problems.
    assert x_rs is None and obj_rs == np.inf, "Expected unbounded result (None, inf)."

    # PuLP formulation
    prob = pulp.LpProblem("UnboundedLP", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    prob += x1 + x2, "Objective"
    prob += x1 - x2 == 0, "EqualityConstraint"
    prob.solve()
    status = pulp.LpStatus[prob.status]
    assert status == "Unbounded" or status == "Undefined", f"Expected unbounded in PuLP, got {status}"


def test_infeasible():
    """
    LP Infeasible:
    Maximize: x1
    Subject to:
      x1 + s1 = -1   (which is infeasible given x1, s1 >= 0)
    The RevisedSimplex should raise a ValueError when validating the basis.
    """
    A = np.array([[1, 1]], dtype=float)
    b = np.array([-1], dtype=float)
    c = np.array([1, 0], dtype=float)
    basis = [1]  # choose the slack as the basis

    try:
        RevisedSimplex(A, b, c, basis)
    except ValueError as e:
        assert "Initial basis is not feasible" in str(e), f"Unexpected error message: {e}"
    else:
        raise AssertionError("Expected ValueError for infeasible initial basis.")


def test_lp7():
    """
    LP7:
    Maximize: 3x1 + 2x2 + x3 + x4
    Subject to:
      x1 + x2 + x3        <= 6
      2x1 + x2      + 3x4 <= 8
           x2 + x3 + x4   <= 5
    Standard form (with slacks s1, s2, s3):
      x1 + x2 + x3      + s1 = 6
      2x1 + x2     + 3x4 + s2 = 8
           x2 + x3 + x4 + s3 = 5
    Basis: [s1, s2, s3] -> indices [4, 5, 6] (with original variables indices 0-3)
    """
    A = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [2, 1, 0, 3, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ], dtype=float)
    b = np.array([6, 8, 5], dtype=float)
    c = np.array([3, 2, 1, 1, 0, 0, 0], dtype=float)
    basis = [4, 5, 6]

    rs_solver = RevisedSimplex(A, b, c, basis)
    x_rs, obj_rs = rs_solver.solve()

    # PuLP formulation
    prob = pulp.LpProblem("LP7", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    x3 = pulp.LpVariable("x3", lowBound=0)
    x4 = pulp.LpVariable("x4", lowBound=0)
    prob += 3 * x1 + 2 * x2 + x3 + x4, "Objective"
    prob += x1 + x2 + x3 <= 6, "Constraint1"
    prob += 2 * x1 + x2 + 3 * x4 <= 8, "Constraint2"
    prob += x2 + x3 + x4 <= 5, "Constraint3"
    prob.solve()

    orig_rs = extract_original_vars(x_rs, 4)
    x1_pulp = pulp.value(x1)
    x2_pulp = pulp.value(x2)
    x3_pulp = pulp.value(x3)
    x4_pulp = pulp.value(x4)
    obj_pulp = pulp.value(prob.objective)

    assert np.isclose(orig_rs[0], x1_pulp, atol=1e-6), f"x1: {orig_rs[0]} vs {x1_pulp}"
    assert np.isclose(orig_rs[1], x2_pulp, atol=1e-6), f"x2: {orig_rs[1]} vs {x2_pulp}"
    assert np.isclose(orig_rs[2], x3_pulp, atol=1e-6), f"x3: {orig_rs[2]} vs {x3_pulp}"
    assert np.isclose(orig_rs[3], x4_pulp, atol=1e-6), f"x4: {orig_rs[3]} vs {x4_pulp}"
    assert np.isclose(obj_rs, obj_pulp, atol=1e-6), f"obj: {obj_rs} vs {obj_pulp}"


if __name__ == '__main__':
    tests = [
        ("LP1", test_lp1),
        ("LP2", test_lp2),
        ("LP3", test_lp3),
        ("LP4 (Degenerate)", test_lp4),
        ("LP5", test_lp5),
        ("Unbounded LP", test_unbounded),
        ("Infeasible LP", test_infeasible),
        ("LP7", test_lp7),
    ]

    print("Running RevisedSimplex tests...\n")
    for name, test in tests:
        run_test(name, test)
