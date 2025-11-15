"""
Utilidades para el controlador: funciones de fuzzificación,
defuzzificación y generación de superficie de control.

Este módulo permite mantener `FuzzyController` más limpio
delegando tareas auxiliares aquí.
"""

from typing import Dict, Tuple
import numpy as np


def fuzzify_inputs(input_variables: Dict, crisp_inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Fuzzifica un diccionario de entradas crisp usando las variables dadas.

    Args:
        input_variables: {nombre: FuzzyVariable}
        crisp_inputs: {nombre: valor_crisp}

    Returns:
        {nombre: {term: membership}}
    """
    input_memberships = {}
    for var_name, crisp_value in crisp_inputs.items():
        if var_name in input_variables:
            fuzzy_var = input_variables[var_name]
            input_memberships[var_name] = fuzzy_var.fuzzify(crisp_value)
        else:
            raise ValueError(f"Variable '{var_name}' no definida")
    return input_memberships


def defuzzify_output(method: str, universe: np.ndarray, aggregated_output: np.ndarray) -> float:
    """Delegador a los métodos de defuzzificación existentes.

    Args:
        method: 'centroid' | 'bisector' | 'mean_of_maximum'
        universe: arreglo del universo de discurso
        aggregated_output: valores agregados de pertenencia

    Returns:
        Valor crisp resultante
    """
    from .defuzzification import Defuzzifier

    defuzz = Defuzzifier()
    if method == 'centroid':
        return defuzz.centroid(universe, aggregated_output)
    elif method == 'bisector':
        return defuzz.bisector(universe, aggregated_output)
    elif method == 'mean_of_maximum':
        return defuzz.mean_of_maximum(universe, aggregated_output)
    else:
        return defuzz.centroid(universe, aggregated_output)


def compute_control_surface(controller, var1, var2, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera la superficie de control delegando a `controller.compute`.

    Args:
        controller: instancia de `FuzzyController`
        var1: primera variable de entrada (FuzzyVariable)
        var2: segunda variable de entrada (FuzzyVariable)
        resolution: puntos por eje

    Returns:
        (X, Y, Z) arrays para graficar
    """
    x_range = np.linspace(var1.universe_range[0], var1.universe_range[1], resolution)
    y_range = np.linspace(var2.universe_range[0], var2.universe_range[1], resolution)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            inputs = {
                var1.name: X[i, j],
                var2.name: Y[i, j]
            }
            Z[i, j] = controller.compute(inputs)

    return X, Y, Z
