"""
Métodos de Defuzzificación
Convierte conjuntos difusos agregados en valores crisp

Implementa los principales métodos:
1. Centroide (Center of Area - COA)
2. Bisector (Bisector of Area - BOA)
3. Mean of Maximum (MOM)
4. Smallest of Maximum (SOM)
5. Largest of Maximum (LOM)

Referencia: Ross, T. J. (2010). Fuzzy Logic with Engineering Applications
"""

import numpy as np
from typing import Optional


class Defuzzifier:
    """Clase para métodos de defuzzificación"""
    
    @staticmethod
    def centroid(universe: np.ndarray, 
                 membership: np.ndarray) -> Optional[float]:
        """
        Método del Centroide (COA - Center of Area)
        
        Es el método más usado. Calcula el centro de gravedad del área
        bajo la curva de la función de pertenencia agregada.
        
        Formula: COA = Σ(x * μ(x)) / Σ(μ(x))
        
        Args:
            universe: Universo de discurso
            membership: Valores de pertenencia agregados
        
        Returns:
            Valor crisp defuzzificado
        """
        numerator = np.sum(universe * membership)
        denominator = np.sum(membership)
        
        if denominator == 0:
            # No hay área bajo la curva - retornar punto medio
            return (universe[0] + universe[-1]) / 2
        
        return numerator / denominator
    
    @staticmethod
    def bisector(universe: np.ndarray, 
                 membership: np.ndarray) -> float:
        """
        Método del Bisector (BOA - Bisector of Area)
        
        Encuentra el valor que divide el área bajo la curva en dos
        partes iguales.
        
        Args:
            universe: Universo de discurso
            membership: Valores de pertenencia agregados
        
        Returns:
            Valor crisp defuzzificado
        """
        total_area = np.sum(membership)
        
        if total_area == 0:
            return (universe[0] + universe[-1]) / 2
        
        half_area = total_area / 2
        cumulative_area = 0
        
        for i, mu in enumerate(membership):
            cumulative_area += mu
            if cumulative_area >= half_area:
                return universe[i]
        
        return universe[-1]
    
    @staticmethod
    def mean_of_maximum(universe: np.ndarray, 
                       membership: np.ndarray) -> float:
        """
        Método de la Media del Máximo (MOM - Mean of Maximum)
        
        Calcula el promedio de todos los puntos donde la función
        de pertenencia alcanza su valor máximo.
        
        Args:
            universe: Universo de discurso
            membership: Valores de pertenencia agregados
        
        Returns:
            Valor crisp defuzzificado
        """
        max_membership = np.max(membership)
        
        if max_membership == 0:
            return (universe[0] + universe[-1]) / 2
        
        # Encontrar todos los índices con valor máximo
        max_indices = np.where(membership == max_membership)[0]
        
        # Promedio de las posiciones con máximo
        return np.mean(universe[max_indices])
    
    @staticmethod
    def smallest_of_maximum(universe: np.ndarray, 
                           membership: np.ndarray) -> float:
        """
        Método del Mínimo del Máximo (SOM - Smallest of Maximum)
        
        Retorna el menor valor donde la función alcanza su máximo.
        
        Args:
            universe: Universo de discurso
            membership: Valores de pertenencia agregados
        
        Returns:
            Valor crisp defuzzificado
        """
        max_membership = np.max(membership)
        
        if max_membership == 0:
            return universe[0]
        
        max_indices = np.where(membership == max_membership)[0]
        return universe[max_indices[0]]
    
    @staticmethod
    def largest_of_maximum(universe: np.ndarray, 
                          membership: np.ndarray) -> float:
        """
        Método del Máximo del Máximo (LOM - Largest of Maximum)
        
        Retorna el mayor valor donde la función alcanza su máximo.
        
        Args:
            universe: Universo de discurso
            membership: Valores de pertenencia agregados
        
        Returns:
            Valor crisp defuzzificado
        """
        max_membership = np.max(membership)
        
        if max_membership == 0:
            return universe[-1]
        
        max_indices = np.where(membership == max_membership)[0]
        return universe[max_indices[-1]]
    
    @staticmethod
    def weighted_average(universe: np.ndarray, 
                        membership: np.ndarray) -> float:
        """
        Método del Promedio Ponderado (simplificado para funciones singleton)
        
        Args:
            universe: Universo de discurso
            membership: Valores de pertenencia agregados
        
        Returns:
            Valor crisp defuzzificado
        """
        return Defuzzifier.centroid(universe, membership)


def compare_defuzzification_methods(universe: np.ndarray, 
                                    membership: np.ndarray) -> dict:
    """
    Compara todos los métodos de defuzzificación
    
    Args:
        universe: Universo de discurso
        membership: Valores de pertenencia agregados
    
    Returns:
        Diccionario con resultados de cada método
    """
    defuzz = Defuzzifier()
    
    results = {
        'centroid': defuzz.centroid(universe, membership),
        'bisector': defuzz.bisector(universe, membership),
        'mean_of_maximum': defuzz.mean_of_maximum(universe, membership),
        'smallest_of_maximum': defuzz.smallest_of_maximum(universe, membership),
        'largest_of_maximum': defuzz.largest_of_maximum(universe, membership),
    }
    
    return results


def visualize_defuzzification(universe: np.ndarray, 
                              membership: np.ndarray,
                              methods: Optional[list] = None):
    """
    Visualiza la función de pertenencia agregada y los puntos de defuzzificación
    
    Args:
        universe: Universo de discurso
        membership: Valores de pertenencia agregados
        methods: Lista de métodos a visualizar (None = todos)
    """
    import matplotlib.pyplot as plt
    
    if methods is None:
        methods = ['centroid', 'bisector', 'mean_of_maximum']
    
    results = compare_defuzzification_methods(universe, membership)
    
    plt.figure(figsize=(12, 6))
    
    # Graficar función de pertenencia agregada
    plt.fill_between(universe, 0, membership, alpha=0.3, color='blue', 
                     label='Función Agregada')
    plt.plot(universe, membership, 'b-', linewidth=2)
    
    # Colores y marcadores para cada método
    colors = {
        'centroid': 'red',
        'bisector': 'green',
        'mean_of_maximum': 'orange',
        'smallest_of_maximum': 'purple',
        'largest_of_maximum': 'brown'
    }
    
    markers = {
        'centroid': 'o',
        'bisector': 's',
        'mean_of_maximum': '^',
        'smallest_of_maximum': 'v',
        'largest_of_maximum': 'd'
    }
    
    labels = {
        'centroid': 'Centroide (COA)',
        'bisector': 'Bisector (BOA)',
        'mean_of_maximum': 'Media del Máximo (MOM)',
        'smallest_of_maximum': 'Mínimo del Máximo (SOM)',
        'largest_of_maximum': 'Máximo del Máximo (LOM)'
    }
    
    # Marcar puntos de defuzzificación
    for method in methods:
        if method in results:
            value = results[method]
            # Encontrar la membresía en ese punto
            idx = np.argmin(np.abs(universe - value))
            mu = membership[idx]
            
            plt.plot(value, mu, markers[method], 
                    color=colors[method], 
                    markersize=12, 
                    label=f"{labels[method]}: {value:.2f}",
                    markeredgecolor='black',
                    markeredgewidth=1.5)
            
            # Línea vertical
            plt.axvline(value, color=colors[method], 
                       linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Universo de Discurso', fontsize=12)
    plt.ylabel('Grado de Pertenencia (μ)', fontsize=12)
    plt.title('Comparación de Métodos de Defuzzificación', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    
    return plt.gcf()


def analyze_defuzzification_differences(universe: np.ndarray, 
                                        membership: np.ndarray):
    """
    Analiza las diferencias entre los métodos de defuzzificación
    """
    results = compare_defuzzification_methods(universe, membership)
    
    print("\n" + "="*70)
    print("ANÁLISIS DE MÉTODOS DE DEFUZZIFICACIÓN")
    print("="*70 + "\n")
    
    # Calcular estadísticas
    values = list(results.values())
    mean_value = np.mean(values)
    std_value = np.std(values)
    
    print(f"{'Método':<30} {'Valor':<12} {'Diferencia vs Media':<20}")
    print("-" * 70)
    
    for method, value in results.items():
        diff = value - mean_value
        diff_pct = (diff / mean_value * 100) if mean_value != 0 else 0
        
        method_name = method.replace('_', ' ').title()
        print(f"{method_name:<30} {value:>8.3f}    {diff:>+8.3f} ({diff_pct:>+6.2f}%)")
    
    print("-" * 70)
    print(f"{'Media':<30} {mean_value:>8.3f}")
    print(f"{'Desviación Estándar':<30} {std_value:>8.3f}")
    print(f"{'Rango (max - min)':<30} {max(values) - min(values):>8.3f}")
    print("\n" + "="*70)
    
    # Recomendaciones
    print("\nRECOMENDACIONES:")
    print("  • Centroide (COA): Método más usado, suave y representativo")
    print("  • Bisector (BOA): Similar al centroide, equilibra áreas")
    print("  • MOM: Enfatiza regiones de máxima certeza")
    print("  • SOM/LOM: Útiles cuando se prefiere comportamiento conservador/agresivo")
    
    return results


if __name__ == "__main__":
    """Prueba de métodos de defuzzificación"""
    
    # Crear una función de pertenencia agregada de ejemplo
    universe = np.linspace(0, 100, 1000)
    
    # Simular salida agregada con dos picos
    membership = np.zeros_like(universe)
    
    # Pico 1: zona baja (20-40)
    peak1_center = 30
    peak1_width = 10
    membership += 0.6 * np.exp(-0.5 * ((universe - peak1_center) / peak1_width) ** 2)
    
    # Pico 2: zona alta (60-80)
    peak2_center = 70
    peak2_width = 8
    membership += 0.9 * np.exp(-0.5 * ((universe - peak2_center) / peak2_width) ** 2)
    
    # Limitar a [0, 1]
    membership = np.minimum(membership, 1.0)
    
    # Analizar diferencias
    results = analyze_defuzzification_differences(universe, membership)
    
    # Visualizar
    import matplotlib.pyplot as plt
    fig = visualize_defuzzification(universe, membership)
    plt.savefig('defuzzification_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado como 'defuzzification_comparison.png'")
    
    # Ejemplo con función simple (un solo pico)
    print("\n" + "="*70)
    print("EJEMPLO CON FUNCIÓN SIMPLE")
    print("="*70)
    
    membership_simple = 0.8 * np.exp(-0.5 * ((universe - 50) / 15) ** 2)
    
    results_simple = compare_defuzzification_methods(universe, membership_simple)
    
    print("\nResultados:")
    for method, value in results_simple.items():
        print(f"  {method.replace('_', ' ').title()}: {value:.2f}")
    
    print("\n✓ Módulo de defuzzificación probado exitosamente")