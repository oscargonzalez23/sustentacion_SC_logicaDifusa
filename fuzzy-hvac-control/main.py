"""
Script Principal - Simulación y Comparación de Controladores
Ejecuta experimentos completos comparando controladores difusos y PID

Autor: [Tu nombre]
Fecha: 2025
Curso: Lógica Difusa en Sistemas de Control
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'fuzzy_controller'))

# Importar módulos propios
from membership_functions import (
    create_temperature_variable,
    create_error_variable,
    create_power_variable
)
from fuzzy_rules import create_hvac_rule_base
from fuzzy_system import (
    FuzzyController,
    PIDController,
    HVACSystem,
    simulate_control,
    calculate_performance_metrics
)


# ==================== CONFIGURACIÓN DE EXPERIMENTOS ====================

class ExperimentConfig:
    """Configuración de experimentos"""
    
    # Parámetros del sistema
    INITIAL_TEMP = 20.0      # °C
    AMBIENT_TEMP = 30.0      # °C
    SETPOINT = 22.0          # °C temperatura deseada
    
    # Parámetros de simulación
    DURATION = 100.0         # minutos
    DT = 0.5                 # paso de tiempo (minutos)
    
    # Parámetros PID (ajustados manualmente)
    PID_KP = 8.0
    PID_KI = 0.3
    PID_KD = 2.0
    
    # Métodos de defuzzificación a probar
    DEFUZZ_METHODS = ['centroid', 'bisector', 'mean_of_maximum']
    
    # Directorio de resultados
    RESULTS_DIR = 'results'


# ==================== EXPERIMENTO 1: COMPARACIÓN BÁSICA ====================

def experiment_1_basic_comparison():
    """
    Experimento 1: Comparación básica entre Fuzzy y PID
    Sin perturbaciones, respuesta a escalón
    """
    print("\n" + "="*70)
    print("EXPERIMENTO 1: Comparación Básica Fuzzy vs PID")
    print("="*70)
    
    # Crear variables difusas
    temp_var = create_temperature_variable()
    error_var = create_error_variable()
    power_var = create_power_variable()
    
    # Crear controlador difuso
    rule_base = create_hvac_rule_base()
    fuzzy_controller = FuzzyController(
        input_variables={'Temperatura': temp_var, 'Error': error_var},
        output_variable=power_var,
        inference_engine=rule_base,
        defuzzification_method='centroid'
    )
    
    # Crear controlador PID
    pid_controller = PIDController(
        Kp=ExperimentConfig.PID_KP,
        Ki=ExperimentConfig.PID_KI,
        Kd=ExperimentConfig.PID_KD,
        output_limits=(0, 100)
    )
    
    # Simular con Fuzzy
    print("\n→ Simulando controlador Difuso...")
    system_fuzzy = HVACSystem(
        initial_temp=ExperimentConfig.INITIAL_TEMP,
        ambient_temp=ExperimentConfig.AMBIENT_TEMP
    )
    results_fuzzy = simulate_control(
        fuzzy_controller,
        system_fuzzy,
        setpoint=ExperimentConfig.SETPOINT,
        duration=ExperimentConfig.DURATION,
        dt=ExperimentConfig.DT
    )
    
    # Simular con PID
    print("→ Simulando controlador PID...")
    system_pid = HVACSystem(
        initial_temp=ExperimentConfig.INITIAL_TEMP,
        ambient_temp=ExperimentConfig.AMBIENT_TEMP
    )
    results_pid = simulate_control(
        pid_controller,
        system_pid,
        setpoint=ExperimentConfig.SETPOINT,
        duration=ExperimentConfig.DURATION,
        dt=ExperimentConfig.DT
    )
    
    # Calcular métricas
    print("\n→ Calculando métricas de desempeño...")
    metrics_fuzzy = calculate_performance_metrics(results_fuzzy)
    metrics_pid = calculate_performance_metrics(results_pid)
    
    # Mostrar resultados
    print("\n" + "-"*70)
    print("MÉTRICAS DE DESEMPEÑO")
    print("-"*70)
    print(f"{'Métrica':<30} {'Fuzzy':<15} {'PID':<15} {'Mejor'}")
    print("-"*70)
    
    comparisons = {
        'Rise Time (min)': ('rise_time', 'menor'),
        'Overshoot (%)': ('overshoot', 'menor'),
        'Settling Time (min)': ('settling_time', 'menor'),
        'Steady-State Error (°C)': ('steady_state_error', 'menor'),
        'IAE': ('IAE', 'menor'),
        'ISE': ('ISE', 'menor'),
        'ITAE': ('ITAE', 'menor')
    }
    
    wins = {'Fuzzy': 0, 'PID': 0}
    
    for metric_name, (key, criterion) in comparisons.items():
        val_fuzzy = metrics_fuzzy.get(key)
        val_pid = metrics_pid.get(key)
        
        if val_fuzzy is not None and val_pid is not None:
            if criterion == 'menor':
                better = 'Fuzzy' if val_fuzzy < val_pid else 'PID'
            else:
                better = 'Fuzzy' if val_fuzzy > val_pid else 'PID'
            
            wins[better] += 1
            better_mark = f"✓ {better}"
        else:
            better_mark = "N/A"
        
        str_fuzzy = f"{val_fuzzy:.3f}" if val_fuzzy is not None else "N/A"
        str_pid = f"{val_pid:.3f}" if val_pid is not None else "N/A"
        
        print(f"{metric_name:<30} {str_fuzzy:<15} {str_pid:<15} {better_mark}")
    
    print("-"*70)
    print(f"GANADOR: {'Fuzzy' if wins['Fuzzy'] > wins['PID'] else 'PID'} "
          f"({wins['Fuzzy']} vs {wins['PID']})")
    print("-"*70)
    
    # Visualizar
    plot_comparison(results_fuzzy, results_pid, 
                   "Experimento 1: Fuzzy vs PID (Sin Perturbaciones)",
                   save_name='exp1_comparison.png')
    
    return results_fuzzy, results_pid, metrics_fuzzy, metrics_pid


# ==================== EXPERIMENTO 2: RESPUESTA A PERTURBACIONES ====================

def experiment_2_disturbances():
    """
    Experimento 2: Respuesta a perturbaciones
    Cambios súbitos en la temperatura
    """
    print("\n" + "="*70)
    print("EXPERIMENTO 2: Respuesta a Perturbaciones")
    print("="*70)
    
    # Crear controladores
    temp_var = create_temperature_variable()
    error_var = create_error_variable()
    power_var = create_power_variable()
    rule_base = create_hvac_rule_base()
    
    fuzzy_controller = FuzzyController(
        input_variables={'Temperatura': temp_var, 'Error': error_var},
        output_variable=power_var,
        inference_engine=rule_base,
        defuzzification_method='centroid'
    )
    
    pid_controller = PIDController(
        Kp=ExperimentConfig.PID_KP,
        Ki=ExperimentConfig.PID_KI,
        Kd=ExperimentConfig.PID_KD,
        output_limits=(0, 100)
    )
    
    # Definir perturbaciones
    disturbances = {
        30.0: +3.0,   # +3°C en t=30min
        60.0: -4.0,   # -4°C en t=60min
    }
    
    print(f"\n→ Perturbaciones programadas:")
    for time, delta in disturbances.items():
        print(f"   t={time}min: {delta:+.1f}°C")
    
    # Simular
    system_fuzzy = HVACSystem(initial_temp=20.0, ambient_temp=30.0)
    results_fuzzy = simulate_control(
        fuzzy_controller, system_fuzzy,
        setpoint=22.0, duration=100.0, dt=0.5,
        disturbances=disturbances
    )
    
    system_pid = HVACSystem(initial_temp=20.0, ambient_temp=30.0)
    results_pid = simulate_control(
        pid_controller, system_pid,
        setpoint=22.0, duration=100.0, dt=0.5,
        disturbances=disturbances
    )
    
    # Visualizar
    plot_comparison(results_fuzzy, results_pid,
                   "Experimento 2: Respuesta a Perturbaciones",
                   save_name='exp2_disturbances.png')
    
    return results_fuzzy, results_pid


# ==================== EXPERIMENTO 3: COMPARACIÓN DE DEFUZZIFICACIÓN ====================

def experiment_3_defuzzification_comparison():
    """
    Experimento 3: Comparar diferentes métodos de defuzzificación
    """
    print("\n" + "="*70)
    print("EXPERIMENTO 3: Comparación de Métodos de Defuzzificación")
    print("="*70)
    
    temp_var = create_temperature_variable()
    error_var = create_error_variable()
    power_var = create_power_variable()
    rule_base = create_hvac_rule_base()
    
    results_by_method = {}
    metrics_by_method = {}
    
    for method in ExperimentConfig.DEFUZZ_METHODS:
        print(f"\n→ Simulando con método: {method}")
        
        fuzzy_controller = FuzzyController(
            input_variables={'Temperatura': temp_var, 'Error': error_var},
            output_variable=power_var,
            inference_engine=rule_base,
            defuzzification_method=method
        )
        
        system = HVACSystem(initial_temp=20.0, ambient_temp=30.0)
        results = simulate_control(
            fuzzy_controller, system,
            setpoint=22.0, duration=100.0, dt=0.5
        )
        
        results_by_method[method] = results
        metrics_by_method[method] = calculate_performance_metrics(results)
    
    # Comparar métricas
    print("\n" + "-"*70)
    print("COMPARACIÓN DE MÉTODOS DE DEFUZZIFICACIÓN")
    print("-"*70)
    print(f"{'Métrica':<25} ", end='')
    for method in ExperimentConfig.DEFUZZ_METHODS:
        print(f"{method[:10]:<15} ", end='')
    print()
    print("-"*70)
    
    metric_keys = ['rise_time', 'overshoot', 'settling_time', 
                   'steady_state_error', 'IAE', 'ISE']
    
    for key in metric_keys:
        print(f"{key.replace('_', ' ').title():<25} ", end='')
        for method in ExperimentConfig.DEFUZZ_METHODS:
            val = metrics_by_method[method].get(key)
            if val is not None:
                print(f"{val:<15.3f} ", end='')
            else:
                print(f"{'N/A':<15} ", end='')
        print()
    
    print("-"*70)
    
    # Visualizar
    plot_defuzzification_comparison(results_by_method,
                                   save_name='exp3_defuzzification.png')
    
    return results_by_method, metrics_by_method


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def plot_comparison(results_fuzzy, results_pid, title, save_name=None):
    """Grafica comparación entre dos controladores"""
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Temperatura
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results_fuzzy['time'], results_fuzzy['temperature'], 
            'b-', linewidth=2, label='Fuzzy', alpha=0.8)
    ax1.plot(results_pid['time'], results_pid['temperature'], 
            'r--', linewidth=2, label='PID', alpha=0.8)
    ax1.axhline(results_fuzzy['setpoint'], color='g', 
               linestyle=':', linewidth=2, label='Setpoint')
    ax1.set_ylabel('Temperatura (°C)', fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Potencia
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(results_fuzzy['time'], results_fuzzy['power'], 
            'b-', linewidth=2, label='Fuzzy', alpha=0.8)
    ax2.plot(results_pid['time'], results_pid['power'], 
            'r--', linewidth=2, label='PID', alpha=0.8)
    ax2.set_ylabel('Potencia HVAC (%)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-5, 105])
    
    # Error
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(results_fuzzy['time'], results_fuzzy['error'], 
            'b-', linewidth=2, label='Fuzzy', alpha=0.8)
    ax3.plot(results_pid['time'], results_pid['error'], 
            'r--', linewidth=2, label='PID', alpha=0.8)
    ax3.axhline(0, color='g', linestyle=':', linewidth=1.5)
    ax3.set_xlabel('Tiempo (minutos)', fontsize=11)
    ax3.set_ylabel('Error (°C)', fontsize=11)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    if save_name:
        os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(ExperimentConfig.RESULTS_DIR, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {filepath}")
    
    plt.close()


def plot_defuzzification_comparison(results_by_method, save_name=None):
    """Grafica comparación de métodos de defuzzificación"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    colors = {'centroid': 'blue', 'bisector': 'red', 'mean_of_maximum': 'green'}
    
    for method, results in results_by_method.items():
        color = colors.get(method, 'black')
        axes[0].plot(results['time'], results['temperature'],
                    linewidth=2, label=method.title(), 
                    color=color, alpha=0.7)
        axes[1].plot(results['time'], results['power'],
                    linewidth=2, label=method.title(),
                    color=color, alpha=0.7)
    
    # Setpoint
    setpoint = results_by_method['centroid']['setpoint']
    axes[0].axhline(setpoint, color='black', 
                   linestyle=':', linewidth=2, label='Setpoint')
    
    axes[0].set_ylabel('Temperatura (°C)', fontsize=11)
    axes[0].set_title('Comparación de Métodos de Defuzzificación', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Tiempo (minutos)', fontsize=11)
    axes[1].set_ylabel('Potencia HVAC (%)', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_name:
        os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(ExperimentConfig.RESULTS_DIR, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {filepath}")
    
    plt.close()


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Ejecuta todos los experimentos"""
    
    print("\n" + "="*70)
    print("SISTEMA DE CONTROL DIFUSO PARA HVAC")
    print("Comparación con Controlador PID Clásico")
    print("="*70)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuración:")
    print(f"  • Temperatura inicial: {ExperimentConfig.INITIAL_TEMP}°C")
    print(f"  • Temperatura ambiente: {ExperimentConfig.AMBIENT_TEMP}°C")
    print(f"  • Setpoint: {ExperimentConfig.SETPOINT}°C")
    print(f"  • Duración: {ExperimentConfig.DURATION} min")
    print(f"  • PID (Kp, Ki, Kd): ({ExperimentConfig.PID_KP}, "
          f"{ExperimentConfig.PID_KI}, {ExperimentConfig.PID_KD})")
    
    # Ejecutar experimentos
    try:
        results1 = experiment_1_basic_comparison()
        results2 = experiment_2_disturbances()
        results3 = experiment_3_defuzzification_comparison()
        
        print("\n" + "="*70)
        print("TODOS LOS EXPERIMENTOS COMPLETADOS EXITOSAMENTE")
        print("="*70)
        print(f"\nResultados guardados en: {ExperimentConfig.RESULTS_DIR}/")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()