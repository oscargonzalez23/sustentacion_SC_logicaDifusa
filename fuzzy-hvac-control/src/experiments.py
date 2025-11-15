from src.fuzzy_controller import (
    FuzzyController,
    create_temperature_variable,
    create_error_variable,
    create_power_variable,
    create_hvac_rule_base,
    calculate_performance_metrics,
)
from src.pid_controller import PIDController
from src.simulation import HVACSystem, simulate_control


def print_metrics_table(metrics_fuzzy, metrics_pid):
    print("\n" + "-"*70)
    print("METRICAS DE DESEMPEÑO")
    print("-"*70)
    print(f"{'Metrica':<30} {'Fuzzy':<15} {'PID':<15} {'Mejor'}")
    print("-"*70)
    
    comparisons = {
        'Rise Time (min)': ('rise_time', 'menor'),
        'Overshoot (%)': ('overshoot', 'menor'),
        'Settling Time (min)': ('settling_time', 'menor'),
        'Steady-State Error (C)': ('steady_state_error', 'menor'),
        'IAE': ('IAE', 'menor'),
        'ISE': ('ISE', 'menor'),
        'ITAE': ('ITAE', 'menor')
    }
    
    wins_fuzzy = 0
    wins_pid = 0
    
    for display_name, (key, comparison) in comparisons.items():
        fuzzy_val = metrics_fuzzy.get(key, float('nan'))
        pid_val = metrics_pid.get(key, float('nan'))
        
        if isinstance(fuzzy_val, (int, float)) and isinstance(pid_val, (int, float)):
            if comparison == 'menor':
                winner = 'Fuzzy' if fuzzy_val < pid_val else 'PID' if pid_val < fuzzy_val else 'Empate'
                if winner == 'Fuzzy':
                    wins_fuzzy += 1
                elif winner == 'PID':
                    wins_pid += 1
            else:
                winner = 'PID' if fuzzy_val < pid_val else 'Fuzzy'
            
            if isinstance(fuzzy_val, float):
                print(f"{display_name:<30} {fuzzy_val:<15.3f} {pid_val:<15.3f} {winner}")
            else:
                print(f"{display_name:<30} {fuzzy_val:<15} {pid_val:<15} {winner}")
        else:
            print(f"{display_name:<30} {str(fuzzy_val):<15} {str(pid_val):<15} N/A")
    
    print("-"*70)
    print(f"GANADOR: {'Fuzzy' if wins_fuzzy > wins_pid else 'PID'} ({wins_fuzzy} vs {wins_pid})")
    print("-"*70)


def run_experiment_1(config):
    print("\n" + "="*70)
    print("EXPERIMENTO 1: Comparacion Basica Fuzzy vs PID")
    print("="*70)
    
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
        Kp=config.PID_KP,
        Ki=config.PID_KI,
        Kd=config.PID_KD,
        output_limits=(0, 100)
    )
    
    print("\n[>] Simulando controlador Difuso...")
    system_fuzzy = HVACSystem(
        initial_temp=config.INITIAL_TEMP,
        ambient_temp=config.AMBIENT_TEMP
    )
    results_fuzzy = simulate_control(
        fuzzy_controller,
        system_fuzzy,
        setpoint=config.SETPOINT,
        duration=config.DURATION,
        dt=config.DT
    )
    
    print("[>] Simulando controlador PID...")
    system_pid = HVACSystem(
        initial_temp=config.INITIAL_TEMP,
        ambient_temp=config.AMBIENT_TEMP
    )
    results_pid = simulate_control(
        pid_controller,
        system_pid,
        setpoint=config.SETPOINT,
        duration=config.DURATION,
        dt=config.DT
    )
    
    print("\n[>] Calculando metricas de desempeño...")
    metrics_fuzzy = calculate_performance_metrics(results_fuzzy)
    metrics_pid = calculate_performance_metrics(results_pid)
    
    print_metrics_table(metrics_fuzzy, metrics_pid)
    
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_fuzzy['time'], results_fuzzy['temperature'], 'b-', linewidth=2)
    ax1.axhline(config.SETPOINT, color='r', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('Temperatura (C)')
    ax1.set_title('Fuzzy - Temperatura')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results_pid['time'], results_pid['temperature'], 'g-', linewidth=2)
    ax2.axhline(config.SETPOINT, color='r', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Temperatura (C)')
    ax2.set_title('PID - Temperatura')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(results_fuzzy['time'], results_fuzzy['temperature'], 'b-', linewidth=2, label='Fuzzy', alpha=0.7)
    ax3.plot(results_pid['time'], results_pid['temperature'], 'g-', linewidth=2, label='PID', alpha=0.7)
    ax3.axhline(config.SETPOINT, color='r', linestyle='--', linewidth=1.5, label='Setpoint')
    ax3.set_ylabel('Temperatura (C)')
    ax3.set_title('Comparacion: Fuzzy vs PID')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(results_fuzzy['time'], results_fuzzy['power'], 'b-', linewidth=2)
    ax4.set_ylabel('Potencia (%)')
    ax4.set_xlabel('Tiempo (min)')
    ax4.set_title('Fuzzy - Control')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(results_pid['time'], results_pid['power'], 'g-', linewidth=2)
    ax5.set_ylabel('Potencia (%)')
    ax5.set_xlabel('Tiempo (min)')
    ax5.set_title('PID - Control')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 100])
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(config.RESULTS_DIR, 'exp1_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico guardado: {filepath}")
    plt.close('all')
    
    return {
        'fuzzy': results_fuzzy,
        'pid': results_pid,
        'metrics_fuzzy': metrics_fuzzy,
        'metrics_pid': metrics_pid
    }


def run_experiment_2(config):
    print("\n" + "="*70)
    print("EXPERIMENTO 2: Respuesta a Perturbaciones")
    print("="*70)
    
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
        Kp=config.PID_KP,
        Ki=config.PID_KI,
        Kd=config.PID_KD,
        output_limits=(0, 100)
    )
    
    disturbances = {
        config.DURATION * 0.3: 3.0,
        config.DURATION * 0.6: -4.0
    }
    
    print("\n[>] Perturbaciones programadas:")
    for t, delta in disturbances.items():
        print(f"   t={t:.1f}min: {delta:+.1f}C")
    
    system_fuzzy = HVACSystem(
        initial_temp=config.INITIAL_TEMP,
        ambient_temp=config.AMBIENT_TEMP
    )
    results_fuzzy = simulate_control(
        fuzzy_controller,
        system_fuzzy,
        setpoint=config.SETPOINT,
        duration=config.DURATION,
        dt=config.DT,
        disturbances=disturbances
    )
    
    system_pid = HVACSystem(
        initial_temp=config.INITIAL_TEMP,
        ambient_temp=config.AMBIENT_TEMP
    )
    results_pid = simulate_control(
        pid_controller,
        system_pid,
        setpoint=config.SETPOINT,
        duration=config.DURATION,
        dt=config.DT,
        disturbances=disturbances
    )
    
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results_fuzzy['time'], results_fuzzy['temperature'], 'b-', linewidth=2, label='Fuzzy', alpha=0.7)
    ax1.plot(results_pid['time'], results_pid['temperature'], 'g-', linewidth=2, label='PID', alpha=0.7)
    ax1.axhline(config.SETPOINT, color='r', linestyle='--', linewidth=1.5, label='Setpoint')
    for t in disturbances.keys():
        ax1.axvline(t, color='orange', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Temperatura (C)')
    ax1.set_title('Respuesta a Perturbaciones')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results_fuzzy['time'], results_fuzzy['power'], 'b-', linewidth=2)
    ax2.set_ylabel('Potencia (%)')
    ax2.set_xlabel('Tiempo (min)')
    ax2.set_title('Fuzzy - Control')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(results_pid['time'], results_pid['power'], 'g-', linewidth=2)
    ax3.set_ylabel('Potencia (%)')
    ax3.set_xlabel('Tiempo (min)')
    ax3.set_title('PID - Control')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(config.RESULTS_DIR, 'exp2_disturbances.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico guardado: {filepath}")
    plt.close('all')
    
    return {
        'fuzzy': results_fuzzy,
        'pid': results_pid
    }


def run_experiment_3(config):
    print("\n" + "="*70)
    print("EXPERIMENTO 3: Comparacion de Metodos de Defuzzificacion")
    print("="*70)
    
    methods = ['centroid', 'bisector', 'mean_of_maximum']
    methods_data = {}
    methods_metrics = {}
    
    for method in methods:
        print(f"\n[>] Simulando con metodo: {method}")
        
        temp_var = create_temperature_variable()
        error_var = create_error_variable()
        power_var = create_power_variable()
        rule_base = create_hvac_rule_base()
        
        fuzzy_controller = FuzzyController(
            input_variables={'Temperatura': temp_var, 'Error': error_var},
            output_variable=power_var,
            inference_engine=rule_base,
            defuzzification_method=method
        )
        
        system = HVACSystem(
            initial_temp=config.INITIAL_TEMP,
            ambient_temp=config.AMBIENT_TEMP
        )
        results = simulate_control(
            fuzzy_controller,
            system,
            setpoint=config.SETPOINT,
            duration=config.DURATION,
            dt=config.DT
        )
        
        methods_data[method] = results
        methods_metrics[method] = calculate_performance_metrics(results)
    
    print("\n" + "-"*70)
    print("COMPARACION DE METODOS DE DEFUZZIFICACION")
    print("-"*70)
    print(f"{'Metrica':<23} {'centroid':<15} {'bisector':<15} {'mean_of_max'}")
    print("-"*70)
    
    for metric_key in ['rise_time', 'overshoot', 'settling_time', 'steady_state_error', 'IAE', 'ISE']:
        metric_name = metric_key.replace('_', ' ').title()
        values = []
        for m in methods:
            val = methods_metrics[m].get(metric_key, float('nan'))
            values.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        print(f"{metric_name:<23} {values[0]:<15} {values[1]:<15} {values[2]}")
    
    print("-"*70)
    
    import matplotlib.pyplot as plt
    import os
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    colors = {'centroid': 'blue', 'bisector': 'red', 'mean_of_maximum': 'green'}
    
    for method in methods:
        color = colors[method]
        axes[0].plot(methods_data[method]['time'], methods_data[method]['temperature'],
                    linewidth=2, label=method, color=color, alpha=0.7)
        axes[1].plot(methods_data[method]['time'], methods_data[method]['power'],
                    linewidth=2, label=method, color=color, alpha=0.7)
    
    axes[0].axhline(config.SETPOINT, color='orange', linestyle='--', linewidth=1.5, label='Setpoint')
    axes[0].set_ylabel('Temperatura (C)')
    axes[0].set_title('Comparacion de Metodos de Defuzzificacion - Temperatura')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Tiempo (min)')
    axes[1].set_ylabel('Potencia (%)')
    axes[1].set_title('Comparacion de Metodos de Defuzzificacion - Control')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    axes[1].legend()
    
    plt.tight_layout()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(config.RESULTS_DIR, 'exp3_defuzzification.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico guardado: {filepath}")
    plt.close('all')
    
    return {
        'methods_data': methods_data,
        'methods_metrics': methods_metrics
    }
