"""
Sistema de Control Difuso Completo (Fuzzy Logic Controller)
Integra: Fuzzificación → Inferencia → Defuzzificación

Este módulo une todos los componentes del controlador difuso
para controlar la temperatura de un sistema HVAC.
"""

import numpy as np
from typing import Dict, Optional
import sys


class FuzzyController:
    """
    Controlador Difuso completo basado en Mamdani
    """
    
    def __init__(self, 
                 input_variables: Dict,
                 output_variable,
                 inference_engine,
                 defuzzification_method: str = 'centroid'):
        """
        Args:
            input_variables: {nombre: FuzzyVariable}
            output_variable: FuzzyVariable de salida
            inference_engine: FuzzyInferenceEngine con reglas
            defuzzification_method: 'centroid', 'bisector', 'mean_of_maximum'
        """
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.inference_engine = inference_engine
        self.defuzzification_method = defuzzification_method
        
        # Historial para análisis
        self.history = {
            'inputs': [],
            'outputs': [],
            'memberships': []
        }
        
    def compute(self, crisp_inputs: Dict[str, float]) -> float:
        """
        Calcula la salida crisp del controlador difuso
        
        Args:
            crisp_inputs: {nombre_variable: valor_crisp}
        
        Returns:
            Salida crisp defuzzificada
        """
        # 1. FUZZIFICACIÓN: Convertir entradas crisp a difusas
        input_memberships = {}
        for var_name, crisp_value in crisp_inputs.items():
            if var_name in self.input_variables:
                fuzzy_var = self.input_variables[var_name]
                input_memberships[var_name] = fuzzy_var.fuzzify(crisp_value)
            else:
                raise ValueError(f"Variable '{var_name}' no definida")
        
        # 2. INFERENCIA: Aplicar reglas difusas
        universe = self.output_variable.get_universe()
        aggregated_output = self.inference_engine.inference(
            input_memberships, 
            self.output_variable
        )
        
        # 3. DEFUZZIFICACIÓN: Convertir salida difusa a crisp
        from defuzzification import Defuzzifier
        defuzz = Defuzzifier()
        
        if self.defuzzification_method == 'centroid':
            crisp_output = defuzz.centroid(universe, aggregated_output)
        elif self.defuzzification_method == 'bisector':
            crisp_output = defuzz.bisector(universe, aggregated_output)
        elif self.defuzzification_method == 'mean_of_maximum':
            crisp_output = defuzz.mean_of_maximum(universe, aggregated_output)
        else:
            crisp_output = defuzz.centroid(universe, aggregated_output)
        
        # Guardar historial
        self.history['inputs'].append(crisp_inputs.copy())
        self.history['outputs'].append(crisp_output)
        self.history['memberships'].append(input_memberships.copy())
        
        return crisp_output
    
    def reset_history(self):
        """Limpia el historial del controlador"""
        self.history = {
            'inputs': [],
            'outputs': [],
            'memberships': []
        }
    
    def get_control_surface(self, 
                           var1_name: str, 
                           var2_name: str,
                           resolution: int = 50) -> tuple:
        """
        Genera superficie de control 3D
        
        Args:
            var1_name: Nombre de la primera variable de entrada
            var2_name: Nombre de la segunda variable de entrada
            resolution: Número de puntos en cada eje
        
        Returns:
            (X, Y, Z) para graficar superficie
        """
        var1 = self.input_variables[var1_name]
        var2 = self.input_variables[var2_name]
        
        x_range = np.linspace(var1.universe_range[0], 
                             var1.universe_range[1], 
                             resolution)
        y_range = np.linspace(var2.universe_range[0], 
                             var2.universe_range[1], 
                             resolution)
        
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                inputs = {
                    var1_name: X[i, j],
                    var2_name: Y[i, j]
                }
                Z[i, j] = self.compute(inputs)
        
        return X, Y, Z


# ==================== CONTROLADOR PID PARA COMPARACIÓN ====================

class PIDController:
    """
    Controlador PID clásico para comparación
    
    Ecuación: u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
    """
    
    def __init__(self, Kp: float, Ki: float, Kd: float,
                 output_limits: tuple = (0, 100)):
        """
        Args:
            Kp: Ganancia proporcional
            Ki: Ganancia integral
            Kd: Ganancia derivativa
            output_limits: (min, max) de la salida
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        
        # Estado interno
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        
        # Historial
        self.history = {
            'errors': [],
            'outputs': [],
            'time': []
        }
        
    def compute(self, error: float, dt: float = 1.0) -> float:
        """
        Calcula salida PID
        
        Args:
            error: Error = Setpoint - Valor_Actual
            dt: Intervalo de tiempo (segundos)
        
        Returns:
            Señal de control
        """
        # Término proporcional
        P = self.Kp * error
        
        # Término integral (anti-windup limitado)
        self.integral += error * dt
        # Anti-windup: limitar integral
        max_integral = self.output_limits[1] / self.Ki if self.Ki != 0 else 1e6
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        I = self.Ki * self.integral
        
        # Término derivativo
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative
        
        # Salida total
        output = P + I + D
        
        # Limitar salida
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Actualizar estado
        self.previous_error = error
        
        # Guardar historial
        self.history['errors'].append(error)
        self.history['outputs'].append(output)
        
        return output
    
    def reset(self):
        """Reinicia el estado del controlador"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        self.history = {
            'errors': [],
            'outputs': [],
            'time': []
        }


# ==================== MODELO DEL SISTEMA HVAC ====================

class HVACSystem:
    """
    Modelo simplificado de un sistema HVAC
    
    Dinámica térmica de primer orden:
    dT/dt = -(T - T_ambient)/τ + gain*power
    
    donde:
    - T: Temperatura actual
    - T_ambient: Temperatura ambiente
    - τ: Constante de tiempo térmica
    - gain: Ganancia del sistema
    - power: Potencia del HVAC (0-100%)
    """
    
    def __init__(self, 
                 initial_temp: float = 20.0,
                 ambient_temp: float = 30.0,
                 time_constant: float = 5.0,
                 gain: float = 0.01):
        """
        Args:
            initial_temp: Temperatura inicial (°C)
            ambient_temp: Temperatura ambiente (°C)
            time_constant: Constante de tiempo τ (minutos)
            gain: Ganancia del sistema (°C por % de potencia)
        """
        self.temperature = initial_temp
        self.ambient_temp = ambient_temp
        self.time_constant = time_constant
        self.gain = gain
        
        # Historial
        self.history = {
            'temperature': [initial_temp],
            'power': [0.0],
            'time': [0.0]
        }
        
    def update(self, power: float, dt: float = 1.0):
        """
        Actualiza la temperatura del sistema
        
        Args:
            power: Potencia del HVAC (0-100%)
            dt: Paso de tiempo (minutos)
        """
        # Ecuación diferencial: dT/dt
        dT_dt = (-(self.temperature - self.ambient_temp) / self.time_constant 
                 + self.gain * power)
        
        # Integración (Euler)
        self.temperature += dT_dt * dt
        
        # Guardar historial
        last_time = self.history['time'][-1]
        self.history['temperature'].append(self.temperature)
        self.history['power'].append(power)
        self.history['time'].append(last_time + dt)
        
        return self.temperature
    
    def add_disturbance(self, delta_temp: float):
        """Añade una perturbación a la temperatura"""
        self.temperature += delta_temp
    
    def set_ambient_temp(self, new_ambient: float):
        """Cambia la temperatura ambiente"""
        self.ambient_temp = new_ambient
    
    def reset(self, initial_temp: float = 20.0):
        """Reinicia el sistema"""
        self.temperature = initial_temp
        self.history = {
            'temperature': [initial_temp],
            'power': [0.0],
            'time': [0.0]
        }


# ==================== FUNCIÓN DE SIMULACIÓN ====================

def simulate_control(controller, 
                    system: HVACSystem,
                    setpoint: float,
                    duration: float = 100.0,
                    dt: float = 1.0,
                    disturbances: Optional[Dict[float, float]] = None):
    """
    Simula el sistema de control en lazo cerrado
    
    Args:
        controller: FuzzyController o PIDController
        system: Sistema HVAC
        setpoint: Temperatura deseada (°C)
        duration: Duración de la simulación (minutos)
        dt: Paso de tiempo (minutos)
        disturbances: {tiempo: cambio_temperatura}
    
    Returns:
        Diccionario con resultados
    """
    if disturbances is None:
        disturbances = {}
    
    time = 0.0
    times = []
    temperatures = []
    powers = []
    errors = []
    
    while time <= duration:
        # Aplicar perturbaciones si existen
        if time in disturbances:
            system.add_disturbance(disturbances[time])
        
        # Calcular error
        current_temp = system.temperature
        error = setpoint - current_temp
        
        # Calcular señal de control
        if isinstance(controller, FuzzyController):
            # Controlador difuso necesita temperatura y error
            power = controller.compute({
                'Temperatura': current_temp,
                'Error': error
            })
        elif isinstance(controller, PIDController):
            # Controlador PID solo necesita error
            power = controller.compute(error, dt)
        else:
            raise ValueError("Tipo de controlador no reconocido")
        
        # Actualizar sistema
        system.update(power, dt)
        
        # Guardar datos
        times.append(time)
        temperatures.append(current_temp)
        powers.append(power)
        errors.append(error)
        
        time += dt
    
    return {
        'time': np.array(times),
        'temperature': np.array(temperatures),
        'power': np.array(powers),
        'error': np.array(errors),
        'setpoint': setpoint
    }


# ==================== MÉTRICAS DE DESEMPEÑO ====================

def calculate_performance_metrics(results: dict) -> dict:
    """
    Calcula métricas de desempeño del controlador
    
    Métricas:
    - Rise time: Tiempo para alcanzar 90% del setpoint
    - Settling time: Tiempo para estabilizarse (±2% del setpoint)
    - Overshoot: Sobrepico máximo
    - Steady-state error: Error en estado estacionario
    - IAE: Integral del error absoluto
    - ISE: Integral del error cuadrático
    """
    time = results['time']
    temp = results['temperature']
    setpoint = results['setpoint']
    error = results['error']
    
    metrics = {}
    
    # Rise time (tiempo para alcanzar 90% del setpoint)
    target_90 = setpoint * 0.9
    rise_indices = np.where(temp >= target_90)[0]
    metrics['rise_time'] = time[rise_indices[0]] if len(rise_indices) > 0 else None
    
    # Overshoot (sobrepico)
    max_temp = np.max(temp)
    metrics['overshoot'] = ((max_temp - setpoint) / setpoint * 100) if setpoint != 0 else 0
    metrics['overshoot'] = max(0, metrics['overshoot'])  # Solo positivo
    
    # Settling time (±2% del setpoint)
    tolerance = 0.02 * setpoint
    settled = np.abs(temp - setpoint) <= tolerance
    
    # Encontrar el último tiempo donde sale de la banda
    if np.any(settled):
        first_settled = np.where(settled)[0][0]
        after_settled = temp[first_settled:]
        if np.all(np.abs(after_settled - setpoint) <= tolerance):
            metrics['settling_time'] = time[first_settled]
        else:
            metrics['settling_time'] = None
    else:
        metrics['settling_time'] = None
    
    # Steady-state error (promedio de los últimos 10 puntos)
    metrics['steady_state_error'] = np.mean(np.abs(error[-10:]))
    
    # IAE: Integral del error absoluto
    metrics['IAE'] = np.trapz(np.abs(error), time)
    
    # ISE: Integral del error cuadrático
    metrics['ISE'] = np.trapz(error**2, time)
    
    # ITAE: Integral del tiempo por error absoluto
    metrics['ITAE'] = np.trapz(time * np.abs(error), time)
    
    return metrics


if __name__ == "__main__":
    print("✓ Sistema de control difuso completo cargado")
    print("\nComponentes disponibles:")
    print("  - FuzzyController: Controlador difuso completo")
    print("  - PIDController: Controlador PID para comparación")
    print("  - HVACSystem: Modelo del sistema térmico")
    print("  - simulate_control(): Función de simulación")
    print("  - calculate_performance_metrics(): Métricas de desempeño")