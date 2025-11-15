"""
Módulo de métricas de desempeño para controladores
"""
import numpy as np
from typing import Dict


def calculate_performance_metrics(results: dict) -> dict:
    """
    Calcula métricas de desempeño del controlador
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
    metrics['overshoot'] = max(0, metrics['overshoot'])

    # Settling time (±2% del setpoint)
    tolerance = 0.02 * setpoint
    settled = np.abs(temp - setpoint) <= tolerance

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
