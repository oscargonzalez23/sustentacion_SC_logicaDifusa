"""
Módulo de Funciones de Pertenencia para Control Difuso
Basado en Zadeh (1965) - Fuzzy Sets Theory

Este módulo implementa diferentes tipos de funciones de pertenencia
para modelar variables lingüísticas en el controlador difuso.
"""

import numpy as np
from typing import Callable, Tuple


class MembershipFunction:
    """Clase base para funciones de pertenencia difusas"""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, x: float) -> float:
        """Evalúa el grado de pertenencia de x"""
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    """
    Función de pertenencia triangular
    Parámetros: [a, b, c] donde b es el pico
    """
    
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        self.a = a
        self.b = b
        self.c = c
        
    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:  # self.b < x < self.c
            return (self.c - x) / (self.c - self.b)


class TrapezoidalMF(MembershipFunction):
    """
    Función de pertenencia trapezoidal
    Parámetros: [a, b, c, d] donde [b, c] es la meseta
    """
    
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        super().__init__(name)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x <= self.c:
            return 1.0
        else:  # self.c < x < self.d
            return (self.d - x) / (self.d - self.c)


class GaussianMF(MembershipFunction):
    """
    Función de pertenencia gaussiana
    Parámetros: mean (centro), sigma (desviación estándar)
    """
    
    def __init__(self, name: str, mean: float, sigma: float):
        super().__init__(name)
        self.mean = mean
        self.sigma = sigma
        
    def evaluate(self, x: float) -> float:
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)


class FuzzyVariable:
    """
    Representa una variable lingüística difusa con múltiples términos
    Ejemplo: Temperatura = {Frío, Templado, Caliente}
    """
    
    def __init__(self, name: str, universe_range: Tuple[float, float]):
        self.name = name
        self.universe_range = universe_range
        self.terms = {}  # Diccionario de términos lingüísticos
        
    def add_term(self, term: MembershipFunction):
        """Añade un término lingüístico a la variable"""
        self.terms[term.name] = term
        
    def fuzzify(self, crisp_value: float) -> dict:
        """
        Fuzzificación: convierte valor crisp a grados de pertenencia
        Returns: dict con {término: grado_de_pertenencia}
        """
        memberships = {}
        for term_name, term_mf in self.terms.items():
            memberships[term_name] = term_mf.evaluate(crisp_value)
        return memberships
    
    def get_universe(self, num_points: int = 1000) -> np.ndarray:
        """Genera el universo de discurso discretizado"""
        return np.linspace(self.universe_range[0], 
                          self.universe_range[1], 
                          num_points)


# ==================== DEFINICIÓN DE VARIABLES LINGÜÍSTICAS ====================

def create_temperature_variable() -> FuzzyVariable:
    """
    Crea la variable lingüística TEMPERATURA
    Rango: 10°C a 35°C
    Términos: Muy Frío, Frío, Templado, Caliente, Muy Caliente
    """
    temp = FuzzyVariable("Temperatura", (10, 35))
    
    # Funciones de pertenencia triangulares
    temp.add_term(TriangularMF("Muy Frío", 10, 10, 15))
    temp.add_term(TriangularMF("Frío", 12, 16, 20))
    temp.add_term(TriangularMF("Templado", 18, 22, 26))
    temp.add_term(TriangularMF("Caliente", 24, 28, 32))
    temp.add_term(TriangularMF("Muy Caliente", 30, 35, 35))
    
    return temp


def create_error_variable() -> FuzzyVariable:
    """
    Crea la variable lingüística ERROR (diferencia con setpoint)
    Rango: -10°C a +10°C
    Términos: Negativo Grande, Negativo, Cero, Positivo, Positivo Grande
    """
    error = FuzzyVariable("Error", (-10, 10))
    
    # Funciones gaussianas para mayor suavidad
    error.add_term(GaussianMF("Negativo Grande", -7, 2))
    error.add_term(GaussianMF("Negativo", -3.5, 1.5))
    error.add_term(GaussianMF("Cero", 0, 1.5))
    error.add_term(GaussianMF("Positivo", 3.5, 1.5))
    error.add_term(GaussianMF("Positivo Grande", 7, 2))
    
    return error


def create_power_variable() -> FuzzyVariable:
    """
    Crea la variable lingüística POTENCIA (salida del controlador)
    Rango: 0% a 100% de potencia del HVAC
    Términos: Muy Baja, Baja, Media, Alta, Muy Alta
    """
    power = FuzzyVariable("Potencia", (0, 100))
    
    # Funciones trapezoidales para zonas de saturación
    power.add_term(TrapezoidalMF("Muy Baja", 0, 0, 10, 20))
    power.add_term(TriangularMF("Baja", 15, 30, 45))
    power.add_term(TriangularMF("Media", 35, 50, 65))
    power.add_term(TriangularMF("Alta", 55, 70, 85))
    power.add_term(TrapezoidalMF("Muy Alta", 80, 90, 100, 100))
    
    return power


# ==================== FUNCIONES DE UTILIDAD ====================

def plot_membership_functions(variable: FuzzyVariable, ax=None):
    """Visualiza las funciones de pertenencia de una variable"""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    universe = variable.get_universe()
    
    for term_name, term_mf in variable.terms.items():
        memberships = [term_mf.evaluate(x) for x in universe]
        ax.plot(universe, memberships, label=term_name, linewidth=2)
    
    ax.set_xlabel(f'{variable.name}')
    ax.set_ylabel('Grado de Pertenencia (μ)')
    ax.set_title(f'Funciones de Pertenencia - {variable.name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    return ax


if __name__ == "__main__":
    """Prueba de las funciones de pertenencia"""
    import matplotlib.pyplot as plt
    
    # Crear variables
    temp = create_temperature_variable()
    error = create_error_variable()
    power = create_power_variable()
    
    # Visualizar
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    plot_membership_functions(temp, axes[0])
    plot_membership_functions(error, axes[1])
    plot_membership_functions(power, axes[2])
    
    plt.tight_layout()
    plt.savefig('membership_functions.png', dpi=300, bbox_inches='tight')
    print("✓ Funciones de pertenencia generadas")
    
    # Ejemplo de fuzzificación
    print("\n=== Ejemplo de Fuzzificación ===")
    temp_actual = 24.5
    fuzzified = temp.fuzzify(temp_actual)
    print(f"Temperatura: {temp_actual}°C")
    for term, membership in fuzzified.items():
        if membership > 0:
            print(f"  {term}: {membership:.3f}")