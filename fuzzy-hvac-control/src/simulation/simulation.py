"""
Modelo del sistema HVAC y función de simulación (módulo separado)
"""
import numpy as np
from typing import Dict, Optional


class HVACSystem:
    """
    Modelo simplificado de un sistema HVAC
    Dinámica térmica de primer orden:
    dT/dt = -(T - T_ambient)/τ + gain*power
    """

    def __init__(self, 
                 initial_temp: float = 20.0,
                 ambient_temp: float = 30.0,
                 time_constant: float = 5.0,
                 gain: float = 0.01):
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
        """Actualiza la temperatura del sistema"""
        dT_dt = (-(self.temperature - self.ambient_temp) / self.time_constant 
                 + self.gain * power)

        self.temperature += dT_dt * dt

        last_time = self.history['time'][-1]
        self.history['temperature'].append(self.temperature)
        self.history['power'].append(power)
        self.history['time'].append(last_time + dt)

        return self.temperature

    def add_disturbance(self, delta_temp: float):
        self.temperature += delta_temp

    def set_ambient_temp(self, new_ambient: float):
        self.ambient_temp = new_ambient

    def reset(self, initial_temp: float = 20.0):
        self.temperature = initial_temp
        self.history = {
            'temperature': [initial_temp],
            'power': [0.0],
            'time': [0.0]
        }


def simulate_control(controller, 
                    system: HVACSystem,
                    setpoint: float,
                    duration: float = 100.0,
                    dt: float = 1.0,
                    disturbances: Optional[Dict[float, float]] = None):
    """
    Simula el sistema de control en lazo cerrado
    """
    if disturbances is None:
        disturbances = {}

    time = 0.0
    times = []
    temperatures = []
    powers = []
    errors = []

    while time <= duration:
        if time in disturbances:
            system.add_disturbance(disturbances[time])

        current_temp = system.temperature
        error = setpoint - current_temp

        # Calcular señal de control
        # Importar localmente para evitar dependencia circular
        from ..pid_controller import PIDController

        if isinstance(controller, PIDController):
            power = controller.compute(error, dt)
        else:
            # Asumir controlador difuso
            power = controller.compute({
                'Temperatura': current_temp,
                'Error': error
            })

        system.update(power, dt)

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
