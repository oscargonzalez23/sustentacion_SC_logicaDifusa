import os
from datetime import datetime
from src.experiments import run_experiment_1, run_experiment_2, run_experiment_3


def input_parameters():
    print("\n" + "="*70)
    print("INGRESE PARAMETROS DE SIMULACION")
    print("="*70)
    print("(Presione ENTER para usar valores por defecto)\n")
    
    def get_float(prompt, default):
        while True:
            try:
                value = input(f"{prompt} [default: {default}]: ").strip()
                if value == "":
                    return default
                return float(value)
            except ValueError:
                print(f"  X Error: Ingrese un numero valido")
    
    print("--- PARAMETROS DEL SISTEMA TERMICO ---")
    initial_temp = get_float("Temperatura inicial (C)", 20.0)
    ambient_temp = get_float("Temperatura ambiente (C)", 30.0)
    setpoint = get_float("Setpoint / Temperatura deseada (C)", 22.0)
    
    print("\n--- PARAMETROS DE SIMULACION ---")
    duration = get_float("Duracion de simulacion (minutos)", 100.0)
    dt = get_float("Paso de tiempo (minutos)", 0.5)
    
    print("\n--- PARAMETROS DEL CONTROLADOR PID ---")
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
    print("\n" + "="*70)
    print("SISTEMA DE CONTROL DIFUSO PARA HVAC")
    print("Comparacion con Controlador PID Clasico")
    print("="*70)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguracion:")
    print(f"  - Temperatura inicial: {ExperimentConfig.INITIAL_TEMP}C")
    print(f"  - Temperatura ambiente: {ExperimentConfig.AMBIENT_TEMP}C")
    print(f"  - Setpoint: {ExperimentConfig.SETPOINT}C")
    print(f"  - Duracion: {ExperimentConfig.DURATION} min")
    print(f"  - Paso de tiempo: {ExperimentConfig.DT} min")
    print(f"  - PID (Kp, Ki, Kd): ({ExperimentConfig.PID_KP}, "
          f"{ExperimentConfig.PID_KI}, {ExperimentConfig.PID_KD})")
    print(f"  - Directorio de resultados: {ExperimentConfig.RESULTS_DIR}\n")


def main():
    params = input_parameters()
    ExperimentConfig.update_from_dict(params)
    
    os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
    
    print_header()
    
    try:
        run_experiment_1(ExperimentConfig)
        run_experiment_2(ExperimentConfig)
        run_experiment_3(ExperimentConfig)
        
        print("\n" + "="*70)
        print("TODOS LOS EXPERIMENTOS COMPLETADOS EXITOSAMENTE")
        print("="*70)
        print(f"Resultados guardados en: {ExperimentConfig.RESULTS_DIR}/\n")
        
    except Exception as e:
        print(f"\nError durante la ejecucion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
