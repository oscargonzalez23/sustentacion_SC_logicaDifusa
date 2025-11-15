import numpy as np


class PIDController:

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 output_limits: tuple = (0, 100)):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

        self.history = {
            'errors': [],
            'outputs': [],
            'time': []
        }

    def compute(self, error: float, dt: float = 1.0) -> float:
        
        P = self.Kp * error

        self.integral += error * dt
        max_integral = self.output_limits[1] / self.Ki if self.Ki != 0 else 1e6
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        I = self.Ki * self.integral

        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative

        output = P + I + D

        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        self.previous_error = error

        self.history['errors'].append(error)
        self.history['outputs'].append(output)

        return output

    def reset(self):
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        self.history = {
            'errors': [],
            'outputs': [],
            'time': []
        }
