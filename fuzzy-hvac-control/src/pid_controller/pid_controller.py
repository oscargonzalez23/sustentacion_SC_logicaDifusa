"""
Controlador PID clásico (módulo separado)
"""
import numpy as np


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
