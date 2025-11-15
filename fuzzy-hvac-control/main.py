"""
Script Principal - Simulación y Comparación de Controladores
Ejecuta experimentos completos comparando controladores difusos y PID

Autor: [Tu nombre]
Fecha: 2025
Curso: Lógica Difusa en Sistemas de Control
"""

import os
from datetime import datetime
from src.experiments import run_experiment_1, run_experiment_2, run_experiment_3


# ==================== CONFIGURACIÓN DE EXPERIMENTOS ====================

def input_parameters():
    """Solicita parámetros de forma interactiva desde consola"""
    print("\n" + "="*70)
    print("INGRESE PARAMETROS DE SIMULACION")
    print("="*70)
    print("(Presione ENTER para usar valores por defecto)\n")
    
    def get_float(prompt, default):
        """Solicita un número flotante con valor por defecto"""
        while True:
            try:
                value = input(f"{prompt} [default: {default}]: ").strip()
                if value == "":
                    return default
                return float(value)
            except ValueError:
                print(f"  ✗ Error: Ingrese un número válido")
    
    # Parámetros del sistema
    print("--- PARÁMETROS DEL SISTEMA TÉRMICO ---")
    initial_temp = get_float("Temperatura inicial (°C)", 20.0)
    ambient_temp = get_float("Temperatura ambiente (°C)", 30.0)
    setpoint = get_float("Setpoint / Temperatura deseada (°C)", 22.0)
    
    # Parámetros de simulación
    print("\n--- PARÁMETROS DE SIMULACIÓN ---")
    duration = get_float("Duración de simulación (minutos)", 100.0)
    dt = get_float("Paso de tiempo (minutos)", 0.5)
    
    # Parámetros PID
    print("\n--- PARÁMETROS DEL CONTROLADOR PID ---")
    kp = get_float("Ganancia proporcional (Kp)", 8.0)
    ki = get_float("Ganancia integral (Ki)", 0.3)
    kd = get_float("Ganancia derivativa (Kd)", 2.0)
    
    print("="*70 + "\n")
    
    return {
        'initial_temp': initial_temp,
        'ambient_temp': ambient_temp,
        'setpoint': setpoint,
        'duration': duration,
        'dt': dt,
        'kp': kp,
        'ki': ki,
        'kd': kd,
        'results': 'results'
    }


class ExperimentConfig:
    """Configuración global de experimentos"""
    
    INITIAL_TEMP = 20.0
    AMBIENT_TEMP = 30.0
    SETPOINT = 22.0
    DURATION = 100.0
    DT = 0.5
    PID_KP = 8.0
    PID_KI = 0.3
    PID_KD = 2.0
    RESULTS_DIR = 'results'
    
    @classmethod
    def update_from_dict(cls, params_dict):
        """Actualiza la configuración desde diccionario de parámetros"""
        cls.INITIAL_TEMP = params_dict['initial_temp']
        cls.AMBIENT_TEMP = params_dict['ambient_temp']
        cls.SETPOINT = params_dict['setpoint']
        cls.DURATION = params_dict['duration']
        cls.DT = params_dict['dt']
        cls.PID_KP = params_dict['kp']
        cls.PID_KI = params_dict['ki']
        cls.PID_KD = params_dict['kd']
        cls.RESULTS_DIR = params_dict['results']


def print_header():
    """Imprime encabezado del programa"""
    print("\n" + "="*70)
    print("SISTEMA DE CONTROL DIFUSO PARA HVAC")
    print("Comparación con Controlador PID Clásico")
    print("="*70)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguración:")
    print(f"  • Temperatura inicial: {ExperimentConfig.INITIAL_TEMP}°C")
    print(f"  • Temperatura ambiente: {ExperimentConfig.AMBIENT_TEMP}°C")
    print(f"  • Setpoint: {ExperimentConfig.SETPOINT}°C")
    print(f"  • Duración: {ExperimentConfig.DURATION} min")
    print(f"  • Paso de tiempo: {ExperimentConfig.DT} min")
    print(f"  • PID (Kp, Ki, Kd): ({ExperimentConfig.PID_KP}, "
          f"{ExperimentConfig.PID_KI}, {ExperimentConfig.PID_KD})")
    print(f"  • Directorio de resultados: {ExperimentConfig.RESULTS_DIR}\n")


def main():
    """Ejecuta todos los experimentos"""
    
    # Solicitar parámetros de forma interactiva
    params = input_parameters()
    ExperimentConfig.update_from_dict(params)
    
    # Crear directorio de resultados
    os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
    
    print_header()
    
    try:
        # Ejecutar los tres experimentos
        run_experiment_1(ExperimentConfig)
        run_experiment_2(ExperimentConfig)
        run_experiment_3(ExperimentConfig)
        
        print("\n" + "="*70)
        print("TODOS LOS EXPERIMENTOS COMPLETADOS EXITOSAMENTE")
        print("="*70)
        print(f"Resultados guardados en: {ExperimentConfig.RESULTS_DIR}/\n")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
