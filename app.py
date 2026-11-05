import ast
import itertools
import math
import operator
import streamlit as st
import numpy as np
import scipy.optimize as opt
import sympy as sp
import matplotlib.pyplot as plt


# MÉTODOS DEL MÓDULO 1: PROGRAMACIÓN LINEAL


def solve_simplex_step_by_step(c, A, b):
    """Resuelve un problema de maximización con restricciones <= por simplex tabla."""
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or len(c) != A.shape[1]:
        raise ValueError("La matriz A debe tener tantas columnas como variables en c.")
    if A.shape[0] != len(b):
        raise ValueError("El número de restricciones debe coincidir con el tamaño de b.")

    m, n = A.shape
    table = np.zeros((m + 1, n + m + 1), dtype=float)
    table[:m, :n] = A
    table[:m, n:n + m] = np.eye(m)
    table[:m, -1] = b
    table[-1, :n] = -c

    initial_table = table.copy()
    iterations = []
    iteration = 1
    max_iter = 50

    while np.any(table[-1, :-1] < -1e-7) and iteration <= max_iter:
        col_pivot = int(np.argmin(table[-1, :-1]))
        ratios = []
        for i in range(m):
            denom = table[i, col_pivot]
            if denom > 1e-7:
                ratios.append(table[i, -1] / denom)
            else:
                ratios.append(np.inf)

        row_pivot = int(np.argmin(ratios))
        if np.isinf(ratios[row_pivot]):
            return {
                "success": False,
                "message": "El problema es no acotado: no hay fila pivote válida.",
                "iterations": iterations,
                "initial_table": initial_table,
                "table": table,
            }

        pivot_value = table[row_pivot, col_pivot]
        table[row_pivot, :] /= pivot_value
        for i in range(m + 1):
            if i != row_pivot:
                factor = table[i, col_pivot]
                if abs(factor) > 1e-7:
                    table[i, :] -= factor * table[row_pivot, :]

        iterations.append({
            "iteracion": iteration,
            "fila_pivote": row_pivot + 1,
            "columna_pivote": col_pivot + 1,
            "razones": [float(value) if np.isfinite(value) else None for value in ratios],
            "elemento_pivote": float(pivot_value),
            "tabla": table.copy(),
        })
        iteration += 1

    if iteration > max_iter and np.any(table[-1, :-1] < -1e-7):
        return {
            "success": False,
            "message": "Se excedió el número máximo de iteraciones.",
            "iterations": iterations,
            "initial_table": initial_table,
            "table": table,
        }

    solution = np.zeros(n, dtype=float)
    for i in range(m):
        pivot_col = np.where(np.abs(table[i, :n]) > 1e-7)[0]
        if len(pivot_col) == 1:
            solution[pivot_col[0]] = table[i, -1]

    return {
        "success": True,
        "message": "Solución óptima encontrada.",
        "objective_value": float(table[-1, -1]),
        "solution": solution,
        "initial_table": initial_table,
        "iterations": iterations,
        "table": table,
    }


def solve_linear_graphical(c, A, b, objective_type):
    """Resuelve y prepara la región factible de un PL de dos variables."""
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(c) != 2:
        raise ValueError("El método gráfico requiere exactamente dos variables.")

    boundaries = [(A[index], b[index]) for index in range(len(b))]
    boundaries.extend([(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)])
    vertices = []
    for (a1, rhs1), (a2, rhs2) in itertools.combinations(boundaries, 2):
        matrix = np.vstack([a1, a2])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        point = np.linalg.solve(matrix, np.array([rhs1, rhs2]))
        if np.all(point >= -1e-7) and np.all(A @ point <= b + 1e-7):
            if not any(np.linalg.norm(point - vertex) < 1e-7 for vertex in vertices):
                vertices.append(point)

    cost = -c if objective_type == "Maximizar" else c
    result = opt.linprog(cost, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    return {
        "vertices": vertices,
        "result": result,
        "objective_value": (
            float(c @ result.x) if result.success else None
        ),
    }


# MÉTODOS DEL MÓDULO 2: PROGRAMACIÓN NO LINEAL


_EXPRESSION_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


def _convert_math_ast(node, variables):
    binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sp.sympify(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
        raise ValueError(f"Nombre no permitido en la expresión: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
        return binary_operators[type(node.op)](
            _convert_math_ast(node.left, variables),
            _convert_math_ast(node.right, variables),
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _convert_math_ast(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _EXPRESSION_FUNCTIONS.get(node.func.id)
        if function is not None and not node.keywords:
            return function(*(_convert_math_ast(arg, variables) for arg in node.args))
    raise ValueError("Expresión no válida. Usa operadores y funciones matemáticas admitidas.")


def parse_math_expression(expression, variable_names):
    """Parsea expresiones matemáticas limitadas sin ejecutar código Python."""
    variables = {name: sp.Symbol(name) for name in variable_names}
    try:
        parsed = ast.parse(expression, mode="eval")
        return _convert_math_ast(parsed.body, variables), tuple(variables.values())
    except (SyntaxError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"No se pudo interpretar la expresión: {exc}") from exc


def parse_nonlinear_constraints(text, variable_names):
    inequalities = []
    equalities = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            comparison = ast.parse(line, mode="eval").body
            if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1:
                raise ValueError("usa una sola comparación por línea")
            left, right = comparison.left, comparison.comparators[0]
            symbols = {name: sp.Symbol(name) for name in variable_names}
            left_expr = _convert_math_ast(left, symbols)
            right_expr = _convert_math_ast(right, symbols)
            difference = left_expr - right_expr
            comparison_op = comparison.ops[0]
            if isinstance(comparison_op, (ast.Lt, ast.LtE)):
                inequalities.append(difference)
            elif isinstance(comparison_op, (ast.Gt, ast.GtE)):
                inequalities.append(-difference)
            elif isinstance(comparison_op, ast.Eq):
                equalities.append(difference)
            else:
                raise ValueError("usa <=, >= o ==.")
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(f"Restricción en línea {line_number}: {exc}") from exc
    return inequalities, equalities

def optimize_nonlinear_function(
    expr_str,
    initial_values,
    method,
    constraints_text="",
    objective_type="Minimizar",
):
    """Resuelve un problema NLP configurable de minimización o maximización."""
    if objective_type not in ("Minimizar", "Maximizar"):
        raise ValueError("El tipo de objetivo debe ser Minimizar o Maximizar.")
    optimization_sign = 1.0 if objective_type == "Minimizar" else -1.0
    variable_names = [f"x{i + 1}" for i in range(len(initial_values))]
    objective_expr, symbols = parse_math_expression(expr_str, variable_names)
    inequalities, equalities = parse_nonlinear_constraints(constraints_text, variable_names)
    function = sp.lambdify(symbols, objective_expr, modules="numpy")
    optimization_expr = optimization_sign * objective_expr
    gradient_expr = [sp.diff(optimization_expr, symbol) for symbol in symbols]
    gradient = sp.lambdify(symbols, gradient_expr, modules="numpy")

    def objective(point):
        return float(function(*point))

    def optimization_objective(point):
        return optimization_sign * objective(point)

    def objective_gradient(point):
        return np.asarray(gradient(*point), dtype=float)

    if method in ("BFGS (numérico)", "Nelder-Mead (numérico)"):
        if inequalities or equalities:
            raise ValueError("BFGS y Nelder-Mead se ofrecen sin restricciones; selecciona SLSQP para restricciones.")
        scipy_method = "BFGS" if method.startswith("BFGS") else "Nelder-Mead"
        iteration_trace = [{
            "iteracion": 0,
            "punto": np.asarray(initial_values, dtype=float).copy(),
            "valor": objective(np.asarray(initial_values, dtype=float)),
        }]

        def record_iteration(point):
            point = np.asarray(point, dtype=float).copy()
            iteration_trace.append({
                "iteracion": len(iteration_trace),
                "punto": point,
                "valor": objective(point),
            })

        result = opt.minimize(
            optimization_objective,
            np.asarray(initial_values, dtype=float),
            jac=objective_gradient if scipy_method == "BFGS" else None,
            method=scipy_method,
            callback=record_iteration,
        )
        return [{
            "success": bool(result.success),
            "message": str(result.message),
            "point": np.asarray(result.x, dtype=float),
            "value": objective(result.x),
            "iterations": int(getattr(result, "nit", 0)),
            "procedure": {
                "kind": "iterations",
                "title": f"Iteraciones de {scipy_method}",
                "method": (
                    "BFGS aproxima el Hessiano inverso y actualiza la dirección de búsqueda."
                    if scipy_method == "BFGS"
                    else "Nelder-Mead transforma y compara los vértices de un simplex en cada iteración."
                ),
                "objective": str(objective_expr),
                "optimization_objective": str(optimization_expr),
                "objective_type": objective_type,
                "rows": iteration_trace,
            },
        }]

    if method == "Descenso de gradiente (iterativo)":
        if inequalities or equalities:
            raise ValueError("El descenso de gradiente implementado es sin restricciones.")
        point = np.asarray(initial_values, dtype=float)
        max_iterations = 2000
        tolerance = 1e-7
        iteration_trace = []
        for iteration in range(1, max_iterations + 1):
            grad = objective_gradient(point)
            if not np.all(np.isfinite(grad)):
                raise ValueError("El gradiente produjo valores no finitos.")
            iteration_trace.append({
                "iteracion": iteration - 1,
                "punto": point.copy(),
                "valor": objective(point),
                "norma_gradiente": float(np.linalg.norm(grad)),
            })
            if np.linalg.norm(grad) < tolerance:
                return [{
                    "success": True,
                    "message": "Convergencia por norma del gradiente.",
                    "point": point,
                    "value": objective(point),
                    "iterations": iteration - 1,
                    "procedure": {
                        "kind": "iterations",
                        "title": "Iteraciones de descenso de gradiente",
                        "method": (
                            "x(k+1) = x(k) - α∇f(x(k)); α se selecciona con búsqueda lineal."
                        ),
                        "objective": str(objective_expr),
                        "objective_type": objective_type,
                        "rows": iteration_trace,
                    },
                }]

            step = 1.0
            current_value = optimization_objective(point)
            while step > 1e-10:
                candidate = point - step * grad
                candidate_value = optimization_objective(candidate)
                if np.isfinite(candidate_value) and candidate_value <= current_value - 1e-4 * step * np.dot(grad, grad):
                    break
                step *= 0.5
            if step <= 1e-10:
                break
            iteration_trace[-1]["paso"] = step
            iteration_trace[-1]["punto_siguiente"] = candidate.copy()
            point = candidate

        return [{
            "success": False,
            "message": "El descenso de gradiente no convergió dentro del límite de iteraciones.",
            "point": point,
            "value": objective(point),
            "iterations": iteration,
            "procedure": {
                "kind": "iterations",
                "title": "Iteraciones de descenso de gradiente",
                "method": (
                    "x(k+1) = x(k) - α∇f(x(k)); α se selecciona con búsqueda lineal."
                ),
                "objective": str(objective_expr),
                "objective_type": objective_type,
                "rows": iteration_trace,
            },
        }]

    if method == "SLSQP (numérico con restricciones)":
        scipy_constraints = []
        for expr in inequalities:
            constraint_func = sp.lambdify(symbols, expr, modules="numpy")
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda point, fn=constraint_func: -float(fn(*point)),
            })
        for expr in equalities:
            constraint_func = sp.lambdify(symbols, expr, modules="numpy")
            scipy_constraints.append({
                "type": "eq",
                "fun": lambda point, fn=constraint_func: float(fn(*point)),
            })
        iteration_trace = [{
            "iteracion": 0,
            "punto": np.asarray(initial_values, dtype=float).copy(),
            "valor": objective(np.asarray(initial_values, dtype=float)),
        }]

        def record_slsqp_iteration(point):
            point = np.asarray(point, dtype=float).copy()
            iteration_trace.append({
                "iteracion": len(iteration_trace),
                "punto": point,
                "valor": objective(point),
            })

        result = opt.minimize(
            optimization_objective,
            np.asarray(initial_values, dtype=float),
            jac=objective_gradient,
            method="SLSQP",
            constraints=scipy_constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
            callback=record_slsqp_iteration,
        )
        return [{
            "success": bool(result.success),
            "message": str(result.message),
            "point": np.asarray(result.x, dtype=float),
            "value": objective(result.x),
            "iterations": int(getattr(result, "nit", 0)),
            "procedure": {
                "kind": "iterations",
                "title": "Iteraciones de SLSQP",
                "method": "SLSQP combina aproximaciones cuadráticas locales con búsqueda lineal y restricciones.",
                "objective": str(objective_expr),
                "optimization_objective": str(optimization_expr),
                "objective_type": objective_type,
                "constraints": [
                    str(expr) + " <= 0" for expr in inequalities
                ] + [str(expr) + " = 0" for expr in equalities],
                "rows": iteration_trace,
            },
        }]

    if method in ("Multiplicadores de Lagrange", "Condiciones KKT"):
        if method == "Multiplicadores de Lagrange" and inequalities:
            raise ValueError("Lagrange admite solo restricciones de igualdad; usa KKT para desigualdades.")
        if method == "Multiplicadores de Lagrange" and not equalities:
            raise ValueError("Agrega al menos una restricción de igualdad para usar Lagrange.")
        if len(inequalities) > 8:
            raise ValueError("KKT admite hasta 8 restricciones de desigualdad para enumerar conjuntos activos.")

        objective_gradient_expr = [sp.diff(optimization_expr, symbol) for symbol in symbols]
        objective_gradient_functions = [
            sp.lambdify(symbols, expr, modules="numpy") for expr in objective_gradient_expr
        ]
        all_candidates = []
        active_sets = (
            [()] if method == "Multiplicadores de Lagrange"
            else [
                subset
                for count in range(len(inequalities) + 1)
                for subset in itertools.combinations(range(len(inequalities)), count)
            ]
        )
        for active_set in active_sets:
            active_inequalities = [inequalities[index] for index in active_set]
            constraint_exprs = equalities + active_inequalities
            multipliers = sp.symbols(f"lambda0:{len(constraint_exprs)}")
            stationarity = [
                objective_gradient_expr[index]
                + sum(
                    multipliers[j] * sp.diff(constraint_exprs[j], symbols[index])
                    for j in range(len(constraint_exprs))
                )
                for index in range(len(symbols))
            ]
            equations = stationarity + constraint_exprs
            unknowns = list(symbols) + list(multipliers)
            equation_functions = [
                sp.lambdify(unknowns, equation, modules="numpy") for equation in equations
            ]

            def residual(values):
                return np.asarray([fn(*values) for fn in equation_functions], dtype=float)

            initial = list(np.asarray(initial_values, dtype=float)) + [0.0] * len(multipliers)
            try:
                root_result = opt.root(residual, initial)
                solution = np.asarray(root_result.x, dtype=float)
                point = solution[:len(symbols)]
                multiplier_values = solution[len(symbols):]
                if not np.all(np.isfinite(solution)) or np.linalg.norm(residual(solution)) > 1e-5:
                    continue
                if any(float(fn(*point)) > 1e-6 for fn in (
                    sp.lambdify(symbols, expr, modules="numpy") for expr in inequalities
                )):
                    continue
                inequality_multipliers = multiplier_values[len(equalities):]
                if method == "Condiciones KKT" and np.any(inequality_multipliers < -1e-6):
                    continue
                if any(np.linalg.norm(point - candidate["point"]) < 1e-5 for candidate in all_candidates):
                    continue
                all_candidates.append({
                    "success": True,
                    "message": "Candidato factible que satisface las ecuaciones de optimalidad.",
                    "point": point,
                    "value": objective(point),
                    "iterations": int(getattr(root_result, "nfev", 0)),
                    "active_constraints": [index + 1 for index in active_set],
                    "procedure": {
                        "kind": "equations",
                        "title": "Sistema de optimalidad y verificación",
                        "method": method,
                        "objective": str(objective_expr),
                        "optimization_objective": str(optimization_expr),
                        "objective_type": objective_type,
                        "equalities": [str(expr) + " = 0" for expr in equalities],
                        "inequalities": [str(expr) + " <= 0" for expr in inequalities],
                        "active_inequalities": [
                            str(expr) + " = 0" for expr in active_inequalities
                        ],
                        "stationarity": [str(expr) + " = 0" for expr in stationarity],
                        "multipliers": [
                            {
                                "name": str(multiplier),
                                "value": float(value),
                                "constraint": str(constraint_exprs[index]),
                            }
                            for index, (multiplier, value) in enumerate(
                                zip(multipliers, multiplier_values)
                            )
                        ],
                        "gradient": [
                            float(function(*point))
                            for function in objective_gradient_functions
                        ],
                        "stationarity_residuals": [
                            float(function(*solution))
                            for function in equation_functions[:len(symbols)]
                        ],
                        "constraint_residuals": [
                            {"restriccion": str(expr), "residuo": float(function(*point))}
                            for expr, function in zip(
                                equalities + inequalities,
                                (
                                    sp.lambdify(symbols, expr, modules="numpy")
                                    for expr in equalities + inequalities
                                ),
                            )
                        ],
                        "complementarity": [
                            {
                                "desigualdad": index + 1,
                                "g(x)": float(
                                    sp.lambdify(symbols, inequalities[index], modules="numpy")(*point)
                                ),
                                "multiplicador": float(multiplier),
                                "lambda*g(x)": float(
                                    multiplier
                                    * sp.lambdify(symbols, inequalities[index], modules="numpy")(*point)
                                ),
                            }
                            for index, multiplier in zip(active_set, inequality_multipliers)
                        ],
                        "active_constraints": [index + 1 for index in active_set],
                    },
                })
            except (FloatingPointError, ValueError, OverflowError):
                continue

        return all_candidates

    raise ValueError("Método de optimización no reconocido.")


# MÉTODOS DEL MÓDULO 3: CONTROL DE INVENTARIOS


def _ensure_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} debe ser mayor que cero.")
    return float(value)


def calculate_eoq_classic(demand, setup_cost, holding_cost):
    D = _ensure_positive(demand, "Demanda")
    S = _ensure_positive(setup_cost, "Costo de preparación")
    H = _ensure_positive(holding_cost, "Costo de mantenimiento")

    q_opt = math.sqrt((2 * D * S) / H)
    n_orders = D / q_opt
    time_between_orders = 365 / n_orders
    total_cost = math.sqrt(2 * D * S * H)
    return {
        "q_opt": q_opt,
        "n_orders": n_orders,
        "time_between_orders": time_between_orders,
        "total_cost": total_cost,
        "procedure": [
            f"Datos: D = {D:g} unidades/año, S = {S:g} por pedido, H = {H:g} por unidad/año.",
            f"Cantidad económica: Q* = √(2DS/H) = √(2·{D:g}·{S:g}/{H:g}) = {q_opt:.4f}.",
            f"Pedidos por año: N = D/Q* = {D:g}/{q_opt:.4f} = {n_orders:.4f}.",
            f"Tiempo entre pedidos: T = 365/N = 365/{n_orders:.4f} = {time_between_orders:.4f} días.",
            f"Costo anual relevante: CT = √(2DSH) = {total_cost:.4f}.",
        ],
    }


def calculate_eoq_shortage(demand, setup_cost, holding_cost, shortage_cost):
    D = _ensure_positive(demand, "Demanda")
    S = _ensure_positive(setup_cost, "Costo de preparación")
    H = _ensure_positive(holding_cost, "Costo de mantenimiento")
    B = _ensure_positive(shortage_cost, "Costo de faltante")

    q_opt = math.sqrt((2 * D * S / H) * ((H + B) / B))
    s_max = math.sqrt((2 * D * S / H) * (B / (H + B)))
    f_max = q_opt - s_max
    total_cost = math.sqrt(2 * D * S * H * (B / (H + B)))
    return {
        "q_opt": q_opt,
        "s_max": s_max,
        "f_max": f_max,
        "total_cost": total_cost,
        "procedure": [
            f"Datos: D = {D:g}, S = {S:g}, H = {H:g}, B = {B:g}.",
            f"Lote óptimo: Q* = √[(2DS/H)((H+B)/B)] = {q_opt:.4f}.",
            f"Inventario máximo: Imax = √[(2DS/H)(B/(H+B))] = {s_max:.4f}.",
            f"Faltante máximo: Fmax = Q* - Imax = {q_opt:.4f} - {s_max:.4f} = {f_max:.4f}.",
            f"Costo anual relevante: CT = √[2DSH(B/(H+B))] = {total_cost:.4f}.",
        ],
    }


# MÉTODOS DEL MÓDULO 4: TEORÍA DE COLAS


def calculate_mm1(lmbda, mu):
    lam = _ensure_positive(lmbda, "Tasa de llegada")
    m = _ensure_positive(mu, "Tasa de servicio")

    if lam >= m:
        raise ValueError("Para M/M/1 la tasa de llegada debe ser menor que la tasa de servicio.")

    rho = lam / m
    lq = (lam ** 2) / (m * (m - lam))
    l = lam / (m - lam)
    wq = lq / lam
    w = l / lam
    p0 = 1 - rho
    return {
        "rho": rho,
        "p0": p0,
        "lq": lq,
        "l": l,
        "wq": wq,
        "w": w,
        "procedure": [
            f"Utilización: ρ = λ/μ = {lam:g}/{m:g} = {rho:.6f}; se verifica ρ < 1.",
            f"Probabilidad de sistema vacío: P0 = 1 - ρ = {p0:.6f}.",
            f"Clientes esperados en cola: Lq = λ²/[μ(μ-λ)] = {lq:.6f}.",
            f"Clientes esperados en el sistema: L = λ/(μ-λ) = {l:.6f}.",
            f"Tiempo de espera en cola (Little): Wq = Lq/λ = {wq:.6f}.",
            f"Tiempo en el sistema (Little): W = L/λ = {w:.6f}.",
        ],
    }


def calculate_mmk(lmbda, mu, servers):
    lam = _ensure_positive(lmbda, "Tasa de llegada")
    m = _ensure_positive(mu, "Tasa de servicio")
    k = int(servers)
    if k < 1:
        raise ValueError("El número de servidores debe ser al menos 1.")

    rho = lam / (k * m)
    if rho >= 1:
        raise ValueError("El sistema es inestable: ρ ≥ 1.")

    sum_terms = sum((lam / m) ** n / math.factorial(n) for n in range(k))
    last_term = ((lam / m) ** k) / (math.factorial(k) * (1 - rho))
    p0 = 1.0 / (sum_terms + last_term)
    lq = (p0 * ((lam / m) ** k) * rho) / (math.factorial(k) * ((1 - rho) ** 2))
    l = lq + (lam / m)
    wq = lq / lam
    w = l / lam
    return {
        "rho": rho,
        "p0": p0,
        "lq": lq,
        "l": l,
        "wq": wq,
        "w": w,
        "procedure": [
            f"Utilización por servidor: ρ = λ/(kμ) = {lam:g}/({k}·{m:g}) = {rho:.6f}; se verifica ρ < 1.",
            f"Normalización: P0 = 1/[Σ(n=0..k-1) (λ/μ)^n/n! + (λ/μ)^k/(k!(1-ρ))] "
            f"= 1/[{sum_terms:.6f} + {last_term:.6f}] = {p0:.6f}.",
            f"Clientes esperados en cola: Lq = P0(λ/μ)^kρ/[k!(1-ρ)²] = {lq:.6f}.",
            f"Clientes esperados en el sistema: L = Lq + λ/μ = {lq:.6f} + {lam / m:.6f} = {l:.6f}.",
            f"Tiempo de espera en cola (Little): Wq = Lq/λ = {wq:.6f}.",
            f"Tiempo en el sistema (Little): W = L/λ = {w:.6f}.",
        ],
        "normalization_terms": [
            {"n": n, "term": (lam / m) ** n / math.factorial(n)}
            for n in range(k)
        ],
    }


def show_procedure(procedure):
    with st.expander("Ver procedimiento de cálculo"):
        if procedure["kind"] in ("steps", "graph"):
            for index, step in enumerate(procedure["steps"], start=1):
                st.markdown(f"{index}. {step}")
            if procedure.get("rows"):
                st.dataframe(procedure["rows"], width="stretch")
            if procedure.get("note"):
                st.caption(procedure["note"])
        elif procedure["kind"] == "iterations":
            st.markdown(f"**{procedure['title']}**")
            for key in ("method", "objective", "optimization_objective"):
                if key in procedure:
                    st.markdown(f"**{key.replace('_', ' ').capitalize()}:** `{procedure[key]}`")
            if procedure.get("constraints"):
                st.markdown("**Restricciones utilizadas:**")
                st.write(procedure["constraints"])
            if procedure.get("objective_type") == "Maximizar":
                st.caption(
                    "El algoritmo minimiza el negativo de f(x) para encontrar el máximo; "
                    "los valores mostrados corresponden a f(x)."
                )
            rows = []
            for entry in procedure["rows"]:
                row = {
                    key: value
                    for key, value in entry.items()
                    if key not in ("punto", "punto_siguiente")
                }
                row.update({
                    f"x{index + 1}": float(value)
                    for index, value in enumerate(entry["punto"])
                })
                if "punto_siguiente" in entry:
                    row.update({
                        f"x{index + 1}_siguiente": float(value)
                        for index, value in enumerate(entry["punto_siguiente"])
                    })
                rows.append(row)
            st.dataframe(rows, width="stretch")
        elif procedure["kind"] == "equations":
            st.markdown(f"**Método:** {procedure['method']}")
            st.markdown(f"**Objetivo original:** `{procedure['objective']}`")
            st.markdown(
                f"**Función optimizada para las condiciones de primer orden:** "
                f"`{procedure['optimization_objective']}`"
            )
            st.markdown("**Gradiente del objetivo transformado en el candidato:**")
            st.write(procedure["gradient"])
            if procedure["equalities"]:
                st.markdown("**Restricciones de igualdad:**")
                st.write(procedure["equalities"])
            if procedure["inequalities"]:
                st.markdown("**Desigualdades normalizadas como g(x) ≤ 0:**")
                st.write(procedure["inequalities"])
            if procedure["active_inequalities"]:
                st.markdown("**Restricciones activas para este candidato:**")
                st.write(procedure["active_inequalities"])
            st.markdown("**Ecuaciones de estacionariedad del Lagrangiano:**")
            st.write(procedure["stationarity"])
            if procedure["multipliers"]:
                st.markdown("**Multiplicadores obtenidos numéricamente:**")
                st.dataframe(procedure["multipliers"], width="stretch")
            st.markdown("**Residuo de estacionariedad:**")
            st.write(procedure["stationarity_residuals"])
            st.markdown("**Residuos de las restricciones:**")
            if procedure["constraint_residuals"]:
                st.dataframe(procedure["constraint_residuals"], width="stretch")
            else:
                st.caption("El modelo no tiene restricciones explícitas.")
            if procedure["method"] == "Condiciones KKT":
                st.markdown("**Verificación de complementariedad:**")
                if procedure["complementarity"]:
                    st.dataframe(procedure["complementarity"], width="stretch")
                else:
                    st.caption("No hay restricciones de desigualdad para comprobar.")
                st.caption(
                    "Se enumeran conjuntos activos; las desigualdades inactivas tienen "
                    "multiplicador cero y se verifica el signo de los multiplicadores activos."
                )
            if procedure["active_constraints"]:
                st.write(
                    "Índices de desigualdades activas: "
                    f"{procedure['active_constraints']} (numeración según el orden ingresado)."
                )


# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Solver de Investigación de Operaciones",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Suite de Investigación de Operaciones")
st.markdown("Herramienta interactiva para resolución de modelos de optimización, inventarios y teoría de colas.")

# Menú de navegación lateral
modulo = st.sidebar.selectbox(
    "Selecciona el Módulo:",
    [
        "1. Programación Lineal",
        "2. Programación No Lineal (NLP)",
        "3. Control de Inventarios (EOQ)",
        "4. Líneas de Espera (Teoría de Colas)"
    ]
)

# ---------------------------------------------------------
# MÓDULO 1: PROGRAMACIÓN LINEAL
# ---------------------------------------------------------
if modulo == "1. Programación Lineal":
    st.header("1. Programación Lineal")
    
    modo = st.radio(
        "Modo de resolución:",
        ["Paso a Paso (Tabla Simplex)", "Solución Rápida (scipy.optimize)", "Método Gráfico (2 variables)"],
    )
    
    col1, col2 = st.columns(2)
    num_vars = col1.number_input("Número de variables de decisión:", min_value=1, max_value=5, value=2)
    num_constraints = col2.number_input("Número de restricciones:", min_value=1, max_value=5, value=2)

    st.subheader("Coeficientes de la Función Objetivo")
    tipo_opt = st.selectbox("Objetivo:", ["Maximizar", "Minimizar"])
    c = []
    cols = st.columns(num_vars)
    for i in range(num_vars):
        c.append(cols[i].number_input(f"x{i+1}", value=1.0, key=f"c_{i}"))

    st.subheader("Restricciones (≤)")
    A = []
    b = []
    for j in range(num_constraints):
        st.write(f"Restricción {j+1}:")
        row_cols = st.columns(num_vars + 1)
        row = []
        for i in range(num_vars):
            row.append(row_cols[i].number_input(f"A[{j+1},{i+1}]", value=1.0, key=f"A_{j}_{i}"))
        rhs = row_cols[-1].number_input(f"Lado derecho b[{j+1}]", value=10.0, key=f"b_{j}")
        A.append(row)
        b.append(rhs)

    if st.button("Resolver modelo"):
        c_arr = np.array(c, dtype=float)
        A_arr = np.array(A, dtype=float)
        b_arr = np.array(b, dtype=float)

        if modo == "Solución Rápida (scipy.optimize)":
            c_cost = -c_arr if tipo_opt == "Maximizar" else c_arr
            res = opt.linprog(c_cost, A_ub=A_arr, b_ub=b_arr, bounds=(0, None), method='highs')

            if res.success:
                z_val = -res.fun if tipo_opt == "Maximizar" else res.fun
                st.success("**Solución Óptima Encontrada**")
                st.metric(label="Valor Óptimo Z", value=f"{z_val:.4f}")
                for i, val in enumerate(res.x):
                    st.write(f"**x{i+1}** = {val:.4f}")
                objective_expression = " + ".join(
                    f"({coefficient:g})x{index + 1}"
                    for index, coefficient in enumerate(c_arr)
                )
                constraint_steps = [
                    f"Restricción {index + 1}: "
                    + " + ".join(
                        f"({coefficient:g})x{variable + 1}"
                        for variable, coefficient in enumerate(row)
                    )
                    + f" ≤ {rhs:g}; holgura = {slack:.6g}."
                    for index, (row, rhs, slack) in enumerate(
                        zip(A_arr, b_arr, b_arr - A_arr @ res.x)
                    )
                ]
                show_procedure({
                    "kind": "steps",
                    "steps": [
                        f"Función objetivo: {'max' if tipo_opt == 'Maximizar' else 'min'} "
                        + f" Z = {objective_expression}.",
                        "Se cambia el signo de los coeficientes del objetivo al maximizar "
                        "para convertir el problema a minimización interna de linprog."
                        if tipo_opt == "Maximizar"
                        else "La función objetivo se resuelve directamente como minimización.",
                        "Se resolvió el modelo con scipy.optimize.linprog, con xᵢ ≥ 0 "
                        "y restricciones de tipo ≤.",
                        "Punto óptimo: "
                        + ", ".join(
                            f"x{index + 1} = {value:.6g}"
                            for index, value in enumerate(res.x)
                        )
                        + f"; valor del objetivo = {z_val:.6g}.",
                        *constraint_steps,
                    ],
                })
            else:
                st.error("No se encontró una solución factible u óptima.")
                show_procedure({
                    "kind": "steps",
                    "steps": [
                        f"El solver terminó con estado {res.status}: {res.message}.",
                        "No se reporta un punto óptimo porque el modelo no produjo una solución óptima finita.",
                    ],
                })

        elif modo == "Método Gráfico (2 variables)":
            if len(c_arr) != 2:
                st.error("El método gráfico está disponible únicamente para dos variables.")
            else:
                result = solve_linear_graphical(c_arr, A_arr, b_arr, tipo_opt)
                vertices = result["vertices"]
                solver_result = result["result"]
                if solver_result.status == 2:
                    st.error("El modelo no tiene región factible.")
                elif solver_result.status == 3:
                    st.warning("La región factible es no acotada en la dirección del objetivo; no existe óptimo finito.")
                elif solver_result.success:
                    st.success(f"Óptimo: Z = {result['objective_value']:.4f}")
                    st.write({f"x{i + 1}": float(value) for i, value in enumerate(solver_result.x)})
                else:
                    st.error(f"No se pudo resolver el modelo: {solver_result.message}")

                if vertices:
                    vertex_rows = [
                        {"x1": float(point[0]), "x2": float(point[1]), "Z": float(c_arr @ point)}
                        for point in vertices
                    ]
                else:
                    vertex_rows = []
                graphical_steps = [
                    f"Se trazan las fronteras de las restricciones: "
                    + "; ".join(
                        f"{row[0]:g}x1 + {row[1]:g}x2 = {rhs:g}"
                        for row, rhs in zip(A_arr, b_arr)
                    )
                    + ", junto con x1 = 0 y x2 = 0.",
                    "Se intersectan las rectas por pares, se descartan intersecciones "
                    "no factibles y se evalúa el objetivo en cada vértice factible.",
                    (
                        f"Óptimo: (x1, x2) = ({solver_result.x[0]:.6g}, "
                        f"{solver_result.x[1]:.6g}), Z = {result['objective_value']:.6g}."
                        if solver_result.success
                        else "No se encontró un óptimo finito; revisar el estado de factibilidad y acotación mostrado arriba."
                    ),
                ]
                show_procedure({
                    "kind": "graph",
                    "steps": graphical_steps,
                    "rows": vertex_rows,
                    "note": "La búsqueda de intersecciones factibles y la selección del óptimo se verifican con el solver lineal.",
                })

                vertex_array = np.asarray(vertices) if vertices else np.empty((0, 2))
                max_x = max(10.0, float(vertex_array[:, 0].max()) * 1.25) if len(vertices) else 10.0
                max_y = max(10.0, float(vertex_array[:, 1].max()) * 1.25) if len(vertices) else 10.0
                x_grid = np.linspace(0, max_x, 250)
                y_grid = np.linspace(0, max_y, 250)
                xx, yy = np.meshgrid(x_grid, y_grid)
                feasible = (xx >= 0) & (yy >= 0)
                for row, rhs in zip(A_arr, b_arr):
                    feasible &= row[0] * xx + row[1] * yy <= rhs + 1e-8

                fig, ax = plt.subplots(figsize=(8, 6))
                if np.all(feasible):
                    ax.set_facecolor("lightgreen")
                elif np.any(feasible):
                    ax.contourf(xx, yy, feasible.astype(int), levels=[0.5, 1.5], alpha=0.25, colors=["lightgreen"])
                for index, (row, rhs) in enumerate(zip(A_arr, b_arr), start=1):
                    if abs(row[1]) > 1e-10:
                        ax.plot(x_grid, (rhs - row[0] * x_grid) / row[1], label=f"Restricción {index}")
                    elif abs(row[0]) > 1e-10:
                        ax.axvline(rhs / row[0], label=f"Restricción {index}")
                if vertices:
                    ax.scatter(vertex_array[:, 0], vertex_array[:, 1], color="darkgreen", label="Vértices factibles")
                if solver_result.success:
                    ax.scatter(solver_result.x[0], solver_result.x[1], color="red", marker="*", s=160, label="Óptimo")
                    optimal_value = result["objective_value"]
                    if abs(c_arr[1]) > 1e-10:
                        ax.plot(
                            x_grid,
                            (optimal_value - c_arr[0] * x_grid) / c_arr[1],
                            color="red",
                            linestyle="--",
                            label="Función objetivo en el óptimo",
                        )
                    elif abs(c_arr[0]) > 1e-10:
                        ax.axvline(
                            optimal_value / c_arr[0],
                            color="red",
                            linestyle="--",
                            label="Función objetivo en el óptimo",
                        )
                ax.set_xlim(0, max_x)
                ax.set_ylim(0, max_y)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_title("Región factible y solución gráfica")
                ax.grid(True, linestyle="--", alpha=0.5)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
                st.pyplot(fig)
        else:
            if tipo_opt == "Minimizar":
                st.warning("El modo paso a paso en esta versión demostrativa soporta optimización de Maximización con restricciones ≤.")
            else:
                result = solve_simplex_step_by_step(c_arr, A_arr, b_arr)
                if not result["success"]:
                    st.error(result["message"])
                    with st.expander("Ver procedimiento Simplex"):
                        st.markdown("**Tabla inicial**")
                        st.dataframe(result["initial_table"])
                        for step in result["iterations"]:
                            st.markdown(f"**Iteración {step['iteracion']}**")
                            st.write(
                                f"Columna pivote {step['columna_pivote']}, fila pivote "
                                f"{step['fila_pivote']}, razones {step['razones']}."
                            )
                            st.dataframe(step["tabla"])
                else:
                    with st.expander("Ver procedimiento Simplex paso a paso"):
                        st.markdown("**Tabla inicial**")
                        st.dataframe(result["initial_table"])
                        for step in result["iterations"]:
                            st.markdown(f"**Iteración {step['iteracion']}**")
                            st.write(
                                f"Entra la variable de la columna {step['columna_pivote']}; "
                                f"sale la variable de la fila {step['fila_pivote']}. "
                                f"Elemento pivote = {step['elemento_pivote']:.6g}."
                            )
                            st.write(
                                "Razones para escoger la fila pivote: "
                                f"{step['razones']}."
                            )
                            st.dataframe(step["tabla"])
                        st.caption(
                            "Se repite el pivoteo hasta que no quedan costos reducidos negativos."
                        )
                    st.success(f"**Valor Óptimo Z** = {result['objective_value']:.4f}")
                    for i, val in enumerate(result["solution"]):
                        st.write(f"**x{i+1}** = {val:.4f}")

# ---------------------------------------------------------
# MÓDULO 2: PROGRAMACIÓN NO LINEAL (NLP)
# ---------------------------------------------------------
elif modulo == "2. Programación No Lineal (NLP)":
    st.header("2. Programación No Lineal (NLP)")
    st.write("Optimización multivariable con métodos numéricos, iterativos y condiciones de optimalidad.")

    objective_type = st.selectbox(
        "Tipo de optimización:",
        ["Minimizar", "Maximizar"],
        key="nlp_objective_type",
    )

    num_nlp_vars = st.number_input("Número de variables:", min_value=1, max_value=10, value=2, key="nlp_num_vars")
    variable_names = [f"x{i + 1}" for i in range(num_nlp_vars)]
    default_objective = (
        "x1**2 + x2**2 - 4*x1 - 6*x2"
        if num_nlp_vars == 2
        else " + ".join(f"{variable}**2" for variable in variable_names)
    )
    expr_str = st.text_input(
        "Función objetivo f(x1, ...):",
        default_objective,
        key=f"nlp_objective_{num_nlp_vars}",
    )
    initial_values = []
    initial_columns = st.columns(min(num_nlp_vars, 5))
    for index, variable in enumerate(variable_names):
        column = initial_columns[index % len(initial_columns)]
        initial_values.append(column.number_input(
            f"Valor inicial {variable}",
            value=0.0,
            key=f"nlp_initial_{index}",
        ))
    constraints_text = st.text_area(
        "Restricciones (una por línea; usa <=, >= o ==):",
        placeholder="x1 + x2 <= 5\nx1 - x2 == 0",
        help="Desigualdades se convierten a g(x) <= 0. Funciones permitidas: sin, cos, tan, exp, log, sqrt y abs.",
    )
    st.caption("Ejemplo de objetivo: `(x1 - 2)**2 + (x2 - 3)**2`. Los métodos KKT/Lagrange encuentran candidatos locales desde el punto inicial.")
    method = st.selectbox(
        "Método:",
        [
            "BFGS (numérico)",
            "Nelder-Mead (numérico)",
            "Descenso de gradiente (iterativo)",
            "SLSQP (numérico con restricciones)",
            "Multiplicadores de Lagrange",
            "Condiciones KKT",
        ],
    )

    if st.button("Resolver NLP"):
        try:
            results = optimize_nonlinear_function(
                expr_str,
                initial_values,
                method,
                constraints_text,
                objective_type,
            )
            if not results:
                st.warning("No se encontraron candidatos KKT/Lagrange factibles desde el punto inicial.")
            else:
                for index, result in enumerate(results, start=1):
                    heading = f"Candidato {index}" if len(results) > 1 else "Resultado"
                    if result["success"]:
                        st.success(f"{heading}: {result['message']}")
                    else:
                        st.warning(f"{heading}: {result['message']}")
                    st.metric("Valor f(x)", f"{result['value']:.6f}")
                    st.write({
                        variable: float(value)
                        for variable, value in zip(variable_names, result["point"])
                    })
                    st.write(f"Iteraciones/evaluaciones: {result['iterations']}")
                    if "active_constraints" in result:
                        st.write(f"Restricciones activas: {result['active_constraints']}")
                    if "procedure" in result:
                        show_procedure(result["procedure"])
        except (ValueError, SyntaxError, TypeError, OverflowError) as e:
            st.error(f"Error al evaluar la expresión matemática: {e}")

# ---------------------------------------------------------
# MÓDULO 3: CONTROL DE INVENTARIOS
# ---------------------------------------------------------
elif modulo == "3. Control de Inventarios (EOQ)":
    st.header("3. Control de Inventarios (Modelo EOQ)")
    
    tipo_eoq = st.selectbox("Selecciona la variante del modelo:", ["EOQ Clásico", "EOQ con Faltantes / Escasez Permitida"])
    
    col1, col2 = st.columns(2)
    D = col1.number_input("Demanda anual (D):", min_value=1.0, value=10000.0)
    S = col2.number_input("Costo de preparación / pedido (S):", min_value=0.1, value=50.0)
    
    col3, col4 = st.columns(2)
    H = col3.number_input("Costo anual de mantener inventario (H):", min_value=0.01, value=2.0)
    
    if tipo_eoq == "EOQ con Faltantes / Escasez Permitida":
        B = col4.number_input("Costo anual de faltante por unidad (B):", min_value=0.01, value=5.0)

    if st.button("Calcular Inventario"):
        try:
            if tipo_eoq == "EOQ Clásico":
                result = calculate_eoq_classic(D, S, H)
                Q = result["q_opt"]
                N = result["n_orders"]
                T = result["time_between_orders"]
                TC = result["total_cost"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Lote Óptimo (Q*)", f"{Q:.2f} unidades")
                c2.metric("N° de Pedidos al año (N)", f"{N:.2f}")
                c3.metric("Tiempo entre pedidos (T)", f"{T:.1f} días")
                st.write(f"**Costo Total Anual de Inventario:** ${TC:.2f}")
                show_procedure({"kind": "steps", "steps": result["procedure"]})

                fig, ax = plt.subplots(figsize=(8, 3))
                t_vals = np.linspace(0, 3, 300)
                q_vals = Q * (1 - (t_vals % 1))
                ax.plot(t_vals, q_vals, color='navy', lw=2)
                ax.set_title("Ciclo de Inventario (Diente de Sierra)")
                ax.set_xlabel("Ciclos")
                ax.set_ylabel("Nivel de Inventario")
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)

            else:
                result = calculate_eoq_shortage(D, S, H, B)
                Q = result["q_opt"]
                S_max = result["s_max"]
                F_max = result["f_max"]
                TC = result["total_cost"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Lote Óptimo (Q*)", f"{Q:.2f} u")
                c2.metric("Inventario Máximo (S*)", f"{S_max:.2f} u")
                c3.metric("Faltante Máximo (F*)", f"{F_max:.2f} u")
                st.write(f"**Costo Total Anual de Inventario:** ${TC:.2f}")
                show_procedure({"kind": "steps", "steps": result["procedure"]})
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------
# MÓDULO 4: LÍNEAS DE ESPERA (COLAS)
# ---------------------------------------------------------
elif modulo == "4. Líneas de Espera (Teoría de Colas)":
    st.header("4. Teoría de Colas (Modelos M/M/1 y M/M/k)")
    
    modelo_cola = st.selectbox("Modelo de cola:", ["M/M/1 (Un solo servidor)", "M/M/k (Múltiples servidores)"])
    
    col1, col2 = st.columns(2)
    lmbda = col1.number_input("Tasa media de llegada (λ - clientes/unidad de tiempo):", min_value=0.01, value=3.0)
    mu = col2.number_input("Tasa media de servicio (μ - clientes/unidad de tiempo):", min_value=0.01, value=5.0)

    if modelo_cola == "M/M/k (Múltiples servidores)":
        k = st.number_input("Número de servidores (k):", min_value=1, value=2)
    else:
        k = 1

    if st.button("Analizar Sistema de Colas"):
        try:
            if k == 1:
                result = calculate_mm1(lmbda, mu)
            else:
                result = calculate_mmk(lmbda, mu, k)

            rho = result["rho"]
            p0 = result["p0"]
            lq = result["lq"]
            l = result["l"]
            wq = result["wq"]
            w = result["w"]

            st.success("Métricas del Sistema Calculadas")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ocupación (ρ)", f"{rho * 100:.1f}%")
            c2.metric("Prob. Sistema Vacío (P0)", f"{p0 * 100:.1f}%")
            c3.metric("N° Medio en Cola (Lq)", f"{lq:.3f} clientes")
            c4.metric("N° Medio en Sistema (L)", f"{l:.3f} clientes")

            c5, c6 = st.columns(2)
            c5.metric("Tiempo Medio en Cola (Wq)", f"{wq:.3f} unidades de tiempo")
            c6.metric("Tiempo Medio en Sistema (W)", f"{w:.3f} unidades de tiempo")
            queue_procedure = {"kind": "steps", "steps": result["procedure"]}
            if "normalization_terms" in result:
                queue_procedure["rows"] = result["normalization_terms"]
                queue_procedure["note"] = (
                    "Cada término es (λ/μ)^n/n! en la suma de normalización de P0."
                )
            show_procedure(queue_procedure)
        except ValueError as e:
            st.error(str(e))