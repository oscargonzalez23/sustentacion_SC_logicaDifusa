"""
Motor de Inferencia Difusa basado en Mamdani
Referencia: Mamdani & Assilian (1975) - Fuzzy Logic Controller

Implementa el proceso de inferencia difusa:
1. Fuzzificación (ya implementada en membership_functions.py)
2. Evaluación de reglas IF-THEN
3. Agregación de reglas
4. Defuzzificación (en defuzzification.py)
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class FuzzyRule:
    """
    Representa una regla difusa del tipo:
    IF <antecedente1> AND <antecedente2> THEN <consecuente>
    
    Ejemplo:
    IF Error es "Positivo Grande" AND Temperatura es "Caliente" 
    THEN Potencia es "Muy Alta"
    """
    antecedents: Dict[str, str]  # {variable_name: term_name}
    consequent: Tuple[str, str]   # (variable_name, term_name)
    weight: float = 1.0           # Peso de la regla (0-1)
    
    def __repr__(self):
        ant_str = " AND ".join([f"{var} es '{term}'" 
                               for var, term in self.antecedents.items()])
        cons_var, cons_term = self.consequent
        return f"IF {ant_str} THEN {cons_var} es '{cons_term}'"


class FuzzyInferenceEngine:
    """
    Motor de inferencia difusa tipo Mamdani
    """
    
    def __init__(self):
        self.rules: List[FuzzyRule] = []
        self.implication_method = "minimum"  # "minimum" o "product"
        
    def add_rule(self, rule: FuzzyRule):
        """Añade una regla a la base de reglas"""
        self.rules.append(rule)
        
    def evaluate_rule(self, rule: FuzzyRule, 
                     input_memberships: Dict[str, Dict[str, float]]) -> float:
        """
        Evalúa el grado de activación de una regla
        
        Args:
            rule: Regla a evaluar
            input_memberships: {variable_name: {term_name: membership}}
        
        Returns:
            Grado de activación de la regla (firing strength)
        """
        # Operador AND: mínimo de las membresías de los antecedentes
        activations = []
        
        for var_name, term_name in rule.antecedents.items():
            if var_name in input_memberships:
                membership = input_memberships[var_name].get(term_name, 0.0)
                activations.append(membership)
            else:
                raise ValueError(f"Variable '{var_name}' no encontrada en inputs")
        
        # T-norma: operador AND difuso (mínimo)
        firing_strength = min(activations) if activations else 0.0
        
        # Aplicar peso de la regla
        return firing_strength * rule.weight
    
    def aggregate_outputs(self, 
                         rule_activations: List[Tuple[FuzzyRule, float]],
                         output_variable) -> np.ndarray:
        """
        Agrega las salidas de todas las reglas activadas
        
        Args:
            rule_activations: Lista de (regla, grado_de_activación)
            output_variable: Variable de salida (FuzzyVariable)
        
        Returns:
            Array con la función de pertenencia agregada
        """
        universe = output_variable.get_universe()
        aggregated = np.zeros_like(universe)
        
        for rule, activation in rule_activations:
            if activation > 0:
                # Obtener el término consecuente
                _, consequent_term = rule.consequent
                consequent_mf = output_variable.terms[consequent_term]
                
                # Evaluar función de pertenencia del consecuente
                consequent_values = np.array([consequent_mf.evaluate(x) 
                                             for x in universe])
                
                # Implicación: recortar (clip) la función de pertenencia
                if self.implication_method == "minimum":
                    implied = np.minimum(consequent_values, activation)
                elif self.implication_method == "product":
                    implied = consequent_values * activation
                else:
                    implied = np.minimum(consequent_values, activation)
                
                # Agregación: máximo de todas las funciones implicadas
                aggregated = np.maximum(aggregated, implied)
        
        return aggregated
    
    def inference(self, 
                  input_memberships: Dict[str, Dict[str, float]],
                  output_variable) -> np.ndarray:
        """
        Proceso completo de inferencia difusa
        
        Args:
            input_memberships: Membresías de las variables de entrada
            output_variable: Variable de salida
        
        Returns:
            Función de pertenencia agregada para la salida
        """
        # Evaluar todas las reglas
        rule_activations = []
        for rule in self.rules:
            activation = self.evaluate_rule(rule, input_memberships)
            if activation > 0:
                rule_activations.append((rule, activation))
        
        # Agregar salidas
        aggregated_output = self.aggregate_outputs(rule_activations, 
                                                   output_variable)
        
        return aggregated_output


# ==================== BASE DE REGLAS PARA CONTROL HVAC ====================

def create_hvac_rule_base() -> FuzzyInferenceEngine:
    """
    Crea la base de reglas para el controlador HVAC
    
    Estrategia de control:
    - Si hace mucho frío y el error es negativo grande → calentar al máximo
    - Si la temperatura es templada y error pequeño → potencia media
    - Si hace mucho calor y error es positivo grande → enfriar al máximo
    
    Total: 25 reglas (5 temperaturas × 5 errores)
    """
    engine = FuzzyInferenceEngine()
    
    # Matriz de reglas basada en conocimiento experto
    # Filas: Error (NG, N, Z, P, PG)
    # Columnas: Temperatura (MF, F, T, C, MC)
    
    rule_matrix = [
        # Error Negativo Grande (temperatura mucho menor que setpoint → calentar)
        [
            ("Muy Frío", "Negativo Grande", "Muy Alta"),    # Hace frío y falta calor
            ("Frío", "Negativo Grande", "Muy Alta"),
            ("Templado", "Negativo Grande", "Alta"),
            ("Caliente", "Negativo Grande", "Media"),
            ("Muy Caliente", "Negativo Grande", "Baja"),
        ],
        # Error Negativo (temperatura menor que setpoint)
        [
            ("Muy Frío", "Negativo", "Alta"),
            ("Frío", "Negativo", "Alta"),
            ("Templado", "Negativo", "Media"),
            ("Caliente", "Negativo", "Baja"),
            ("Muy Caliente", "Negativo", "Muy Baja"),
        ],
        # Error Cero (temperatura en setpoint → mantener)
        [
            ("Muy Frío", "Cero", "Media"),
            ("Frío", "Cero", "Media"),
            ("Templado", "Cero", "Media"),
            ("Caliente", "Cero", "Media"),
            ("Muy Caliente", "Cero", "Media"),
        ],
        # Error Positivo (temperatura mayor que setpoint → enfriar)
        [
            ("Muy Frío", "Positivo", "Muy Alta"),
            ("Frío", "Positivo", "Alta"),
            ("Templado", "Positivo", "Media"),
            ("Caliente", "Positivo", "Baja"),
            ("Muy Caliente", "Positivo", "Muy Baja"),
        ],
        # Error Positivo Grande (temperatura mucho mayor que setpoint)
        [
            ("Muy Frío", "Positivo Grande", "Media"),
            ("Frío", "Positivo Grande", "Baja"),
            ("Templado", "Positivo Grande", "Baja"),
            ("Caliente", "Positivo Grande", "Muy Baja"),
            ("Muy Caliente", "Positivo Grande", "Muy Baja"),
        ],
    ]
    
    error_terms = ["Negativo Grande", "Negativo", "Cero", "Positivo", "Positivo Grande"]
    
    # Generar reglas desde la matriz
    for i, error_term in enumerate(error_terms):
        for temp_term, err_term, power_term in rule_matrix[i]:
            rule = FuzzyRule(
                antecedents={
                    "Temperatura": temp_term,
                    "Error": err_term
                },
                consequent=("Potencia", power_term),
                weight=1.0
            )
            engine.add_rule(rule)
    
    return engine


def create_simplified_rule_base() -> FuzzyInferenceEngine:
    """
    Base de reglas simplificada (9 reglas) para pruebas rápidas
    Solo usa Error como entrada
    """
    engine = FuzzyInferenceEngine()
    
    # Reglas basadas principalmente en el error
    rules = [
        FuzzyRule(
            antecedents={"Error": "Negativo Grande"},
            consequent=("Potencia", "Muy Alta"),
        ),
        FuzzyRule(
            antecedents={"Error": "Negativo"},
            consequent=("Potencia", "Alta"),
        ),
        FuzzyRule(
            antecedents={"Error": "Cero"},
            consequent=("Potencia", "Media"),
        ),
        FuzzyRule(
            antecedents={"Error": "Positivo"},
            consequent=("Potencia", "Baja"),
        ),
        FuzzyRule(
            antecedents={"Error": "Positivo Grande"},
            consequent=("Potencia", "Muy Baja"),
        ),
    ]
    
    for rule in rules:
        engine.add_rule(rule)
    
    return engine


# ==================== FUNCIONES DE ANÁLISIS ====================

def print_rule_base(engine: FuzzyInferenceEngine):
    """Imprime todas las reglas de forma legible"""
    print(f"\n{'='*70}")
    print(f"BASE DE REGLAS DIFUSAS ({len(engine.rules)} reglas)")
    print(f"{'='*70}\n")
    
    for i, rule in enumerate(engine.rules, 1):
        print(f"R{i:02d}: {rule}")


def visualize_rule_activation(engine: FuzzyInferenceEngine,
                              input_memberships: Dict[str, Dict[str, float]]):
    """
    Visualiza qué reglas se activan y con qué fuerza
    """
    print(f"\n{'='*70}")
    print("REGLAS ACTIVADAS")
    print(f"{'='*70}\n")
    
    activations = []
    for i, rule in enumerate(engine.rules, 1):
        activation = engine.evaluate_rule(rule, input_memberships)
        if activation > 0.01:  # Umbral mínimo
            activations.append((i, rule, activation))
    
    # Ordenar por fuerza de activación
    activations.sort(key=lambda x: x[2], reverse=True)
    
    for rule_num, rule, activation in activations:
        bar = '█' * int(activation * 50)
        print(f"R{rule_num:02d} [{activation:5.3f}] {bar}")
        print(f"     {rule}\n")
    
    if not activations:
        print("⚠ Ninguna regla activada")


if __name__ == "__main__":
    """Prueba del motor de inferencia"""
    
    # Crear base de reglas
    engine = create_hvac_rule_base()
    print_rule_base(engine)
    
    # Simular una entrada
    print("\n" + "="*70)
    print("EJEMPLO DE INFERENCIA")
    print("="*70)
    
    # Suponer: Temperatura = 26°C, Error = +3°C (hace calor, debe enfriar)
    input_memberships = {
        "Temperatura": {
            "Muy Frío": 0.0,
            "Frío": 0.0,
            "Templado": 0.4,
            "Caliente": 0.6,
            "Muy Caliente": 0.0
        },
        "Error": {
            "Negativo Grande": 0.0,
            "Negativo": 0.0,
            "Cero": 0.2,
            "Positivo": 0.8,
            "Positivo Grande": 0.0
        }
    }
    
    print("\nEntradas:")
    print("  Temperatura: Templado(0.4), Caliente(0.6)")
    print("  Error: Cero(0.2), Positivo(0.8)")
    
    visualize_rule_activation(engine, input_memberships)
    
    print("\n✓ Motor de inferencia probado exitosamente")