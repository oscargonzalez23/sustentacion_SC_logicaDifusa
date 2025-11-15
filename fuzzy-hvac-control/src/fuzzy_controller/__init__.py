"""
Paquete fuzzy_controller: Controlador difuso basado en Mamdani.

Módulos:
  - membership_functions: Definición de funciones de pertenencia (triangular, trapezoidal, gaussiana)
  - fuzzy_rules: Motor de inferencia difusa y base de reglas
  - fuzzy_system: Controlador difuso completo
  - defuzzification: Métodos de defuzzificación (centroide, bisector, media del máximo)
  - controller_utils: Utilidades (fuzzificación, defuzzificación, superficie de control)
  - metrics: Cálculo de métricas de desempeño
"""

from .fuzzy_system import FuzzyController
from .membership_functions import (
    create_temperature_variable,
    create_error_variable,
    create_power_variable,
    FuzzyVariable,
)
from .fuzzy_rules import create_hvac_rule_base, FuzzyInferenceEngine
from .controller_utils import fuzzify_inputs, defuzzify_output, compute_control_surface
from .metrics import calculate_performance_metrics

__all__ = [
    'FuzzyController',
    'create_temperature_variable',
    'create_error_variable',
    'create_power_variable',
    'FuzzyVariable',
    'create_hvac_rule_base',
    'FuzzyInferenceEngine',
    'fuzzify_inputs',
    'defuzzify_output',
    'compute_control_surface',
    'calculate_performance_metrics',
]
