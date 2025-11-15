import numpy as np
from typing import Dict, Optional

from .controller_utils import (
    fuzzify_inputs,
    defuzzify_output,
    compute_control_surface
)


class FuzzyController:
    
    def __init__(self, 
                 input_variables: Dict,
                 output_variable,
                 inference_engine,
                 defuzzification_method: str = 'centroid'):
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.inference_engine = inference_engine
        self.defuzzification_method = defuzzification_method
        
        self.history = {
            'inputs': [],
            'outputs': [],
            'memberships': []
        }
        
    def compute(self, crisp_inputs: Dict[str, float]) -> float:
        
        input_memberships = fuzzify_inputs(self.input_variables, crisp_inputs)

        universe = self.output_variable.get_universe()
        aggregated_output = self.inference_engine.inference(
            input_memberships,
            self.output_variable
        )

        crisp_output = defuzzify_output(self.defuzzification_method, universe, aggregated_output)
        
        self.history['inputs'].append(crisp_inputs.copy())
        self.history['outputs'].append(crisp_output)
        self.history['memberships'].append(input_memberships.copy())
        
        return crisp_output
    
    def reset_history(self):
        self.history = {
            'inputs': [],
            'outputs': [],
            'memberships': []
        }
    
    def get_control_surface(self, 
                           var1_name: str, 
                           var2_name: str,
                           resolution: int = 50) -> tuple:
        
        var1 = self.input_variables[var1_name]
        var2 = self.input_variables[var2_name]

        return compute_control_surface(self, var1, var2, resolution)


# Las implementaciones de PID, HVACSystem, simulación y métricas
# fueron movidas a módulos separados para mantener este archivo
# liviano. Reexportamos los nombres para compatibilidad.

# Nota: los módulos están en el mismo paquete `src/fuzzy_controller/`
# por lo tanto se importan directamente.



if __name__ == "__main__":
    print("✓ Sistema de control difuso completo cargado")
    print("\nComponentes disponibles:")
    print("  - FuzzyController: Controlador difuso completo")
    print("  - PIDController: Controlador PID para comparación (en pid_controller.py)")
    print("  - HVACSystem: Modelo del sistema térmico (en simulation.py)")
    print("  - simulate_control(): Función de simulación (en simulation.py)")
    print("  - calculate_performance_metrics(): Métricas de desempeño (en metrics.py)")