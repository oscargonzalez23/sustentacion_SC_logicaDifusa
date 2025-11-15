"""
Utilidades para visualización y generación de gráficos
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def save_and_close_figure(filename, results_dir='results'):
    """Guarda figura, limpia la memoria y cierra la figura"""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {filepath}")
    plt.close('all')  # Cierra todos los gráficos


def create_comparison_plot(time_fuzzy, temp_fuzzy, time_pid, temp_pid, 
                          setpoint, power_fuzzy, power_pid):
    """Crea gráfico de comparación Fuzzy vs PID"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Temperatura - Fuzzy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_fuzzy, temp_fuzzy, 'b-', linewidth=2, label='Temperatura')
    ax1.axhline(setpoint, color='r', linestyle='--', linewidth=1.5, label='Setpoint')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Controlador Difuso - Temperatura')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Subplot 2: Temperatura - PID
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_pid, temp_pid, 'g-', linewidth=2, label='Temperatura')
    ax2.axhline(setpoint, color='r', linestyle='--', linewidth=1.5, label='Setpoint')
    ax2.set_ylabel('Temperatura (°C)')
    ax2.set_title('Controlador PID - Temperatura')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Comparación de temperaturas
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time_fuzzy, temp_fuzzy, 'b-', linewidth=2, label='Fuzzy', alpha=0.7)
    ax3.plot(time_pid, temp_pid, 'g-', linewidth=2, label='PID', alpha=0.7)
    ax3.axhline(setpoint, color='r', linestyle='--', linewidth=1.5, label='Setpoint')
    ax3.set_ylabel('Temperatura (°C)')
    ax3.set_title('Comparación: Fuzzy vs PID')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Subplot 4: Potencia - Fuzzy
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_fuzzy, power_fuzzy, 'b-', linewidth=2)
    ax4.set_ylabel('Potencia (%)')
    ax4.set_xlabel('Tiempo (min)')
    ax4.set_title('Controlador Difuso - Potencia')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    # Subplot 5: Potencia - PID
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_pid, power_pid, 'g-', linewidth=2)
    ax5.set_ylabel('Potencia (%)')
    ax5.set_xlabel('Tiempo (min)')
    ax5.set_title('Controlador PID - Potencia')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 100])
    
    return fig


def create_disturbance_plot(time, temp_fuzzy, temp_pid, power_fuzzy, 
                           power_pid, setpoint, disturbances):
    """Crea gráfico de respuesta a perturbaciones"""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Temperatura
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, temp_fuzzy, 'b-', linewidth=2, label='Fuzzy', alpha=0.7)
    ax1.plot(time, temp_pid, 'g-', linewidth=2, label='PID', alpha=0.7)
    ax1.axhline(setpoint, color='r', linestyle='--', linewidth=1.5, label='Setpoint')
    
    # Marcar perturbaciones
    for disturb_time, disturb_value in disturbances:
        ax1.axvline(disturb_time, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax1.text(disturb_time, ax1.get_ylim()[1]*0.95, f'{disturb_value:+.1f}°C', 
                ha='center', fontsize=9, color='orange')
    
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Respuesta a Perturbaciones de Temperatura Ambiente')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Potencia Fuzzy
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, power_fuzzy, 'b-', linewidth=2)
    ax2.set_ylabel('Potencia (%)')
    ax2.set_xlabel('Tiempo (min)')
    ax2.set_title('Controlador Difuso - Potencia')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # Potencia PID
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, power_pid, 'g-', linewidth=2)
    ax3.set_ylabel('Potencia (%)')
    ax3.set_xlabel('Tiempo (min)')
    ax3.set_title('Controlador PID - Potencia')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    return fig


def create_defuzzification_plot(time_results, methods_data, setpoint):
    """Crea gráfico comparativo de métodos de defuzzificación"""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {'centroid': 'blue', 'bisector': 'green', 'mean_of_maximum': 'red'}
    
    # Gráfico principal de temperaturas
    ax1 = fig.add_subplot(gs[0, :])
    for method, data in methods_data.items():
        ax1.plot(data['time'], data['temperature'], 
                label=method, linewidth=2, color=colors.get(method, 'black'), alpha=0.7)
    ax1.axhline(setpoint, color='orange', linestyle='--', linewidth=1.5, label='Setpoint')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_title('Comparación de Métodos de Defuzzificación - Temperatura')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráficos individuales de control
    for idx, (method, data) in enumerate(methods_data.items(), 1):
        ax = fig.add_subplot(gs[1, (idx-1) % 2])
        ax.plot(data['time'], data['power'], linewidth=2, color=colors[method])
        ax.set_ylabel('Potencia (%)')
        ax.set_xlabel('Tiempo (min)')
        ax.set_title(f'Método: {method} - Control')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        if idx == 3:
            break
    
    return fig
