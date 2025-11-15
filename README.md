# Control Difuso para Sistema HVAC

## Descripcion

Sistema de control basado en logica difusa (Fuzzy Logic) para sistemas de climatizacion (HVAC). El proyecto compara el desempeño de un controlador difuso de inferencia Mamdani contra un controlador PID clasico mediante simulaciones de laboratorio.

## Caracteristicas

- **Controlador Fuzzy**: Implementacion de sistema de inferencia difusa con fuzzificacion, inferencia de reglas y defuzzificacion
- **Controlador PID**: Implementacion clasica de PID con anti-windup para comparacion
- **Modelo HVAC**: Dinamica termica de primer orden que simula comportamiento del sistema de climatizacion
- **Metricas de Desempeño**: Calculo automatico de rise time, overshoot, steady-state error, IAE, ISE, ITAE
- **Tres Experimentos**:
  1. Comparacion basica Fuzzy vs PID sin perturbaciones
  2. Respuesta a perturbaciones ambientales
  3. Comparacion de metodos de defuzzificacion (centroid, bisector, mean_of_maximum)

## Estructura del Proyecto

```
fuzzy-hvac-control/
├── main.py                          # Punto de entrada principal
├── README.md                        # Este archivo
├── test_input.txt                   # Archivo con parametros de entrada
├── test_input_simple.txt            # Entrada simplificada
├── results/                         # Directorio de salida (graficos)
└── src/
    ├── __init__.py
    ├── experiments.py               # Logica de los 3 experimentos
    ├── visualization.py             # Funciones de graficacion
    ├── fuzzy_controller/
    │   ├── __init__.py
    │   ├── fuzzy_system.py          # Clase FuzzyController
    │   ├── membership_functions.py  # Funciones de pertenencia
    │   ├── fuzzy_rules.py           # Base de reglas difusas
    │   ├── defuzzification.py       # Metodos de defuzzificacion
    │   ├── controller_utils.py      # Utilidades de control
    │   └── metrics.py               # Calculo de metricas
    ├── pid_controller/
    │   ├── __init__.py
    │   └── pid_controller.py        # Clase PIDController
    └── simulation/
        ├── __init__.py
        └── simulation.py            # Clase HVACSystem y simulacion
```

## Requisitos

- Python 3.7+
- numpy
- matplotlib

Instalar dependencias:
```bash
pip install numpy matplotlib
```

## Uso

### Opcion 1: Entrada Interactiva desde Consola

Ejecutar el programa y seguir los prompts interactivos:

```bash
python main.py
```

El programa solicitara:
- Temperatura inicial (°C) [default: 20.0]
- Temperatura ambiente (°C) [default: 30.0]
- Setpoint / Temperatura deseada (°C) [default: 22.0]
- Duracion de simulacion (minutos) [default: 100.0]
- Paso de tiempo (minutos) [default: 0.5]
- Ganancia proporcional (Kp) [default: 8.0]
- Ganancia integral (Ki) [default: 0.3]
- Ganancia derivativa (Kd) [default: 2.0]

Presionar ENTER para usar valores por defecto.

### Opcion 2: Entrada desde Archivo

Crear archivo con parametros (uno por linea) y ejecutar con redirección:

```bash
python main.py < test_input.txt
```

Formato del archivo (8 lineas):
```
20.0          # Temperatura inicial
30.0          # Temperatura ambiente
22.0          # Setpoint
100.0         # Duracion simulacion
0.5           # Paso de tiempo
8.0           # Kp
0.3           # Ki
2.0           # Kd
```

Archivos de ejemplo incluidos:
- `test_input.txt`: Parametros con valores por defecto
- `test_input_simple.txt`: Parametros alternativos

### Opcion 3: Ejecutar Interactivamente

Simplemente ingresar valores cuando el programa lo solicite:

```
======================================================================
INGRESE PARAMETROS DE SIMULACION
======================================================================
(Presione ENTER para usar valores por defecto)

--- PARAMETROS DEL SISTEMA TERMICO ---
Temperatura inicial (C) [default: 20.0]: 18.5
Temperatura ambiente (C) [default: 30.0]: 32.0
Setpoint / Temperatura deseada (C) [default: 22.0]: 24.0
...
```

## Salida

El programa genera tres graficos PNG guardados en la carpeta `results/`:

1. **exp1_comparison.png**: Comparacion basica Fuzzy vs PID
   - Temperatura vs Tiempo (Fuzzy)
   - Temperatura vs Tiempo (PID)
   - Comparacion superpuesta con setpoint
   - Señales de control de ambos controladores

2. **exp2_disturbances.png**: Respuesta a perturbaciones
   - Temperatura con cambios ambientales repentinos
   - Lineas verticales marcan momentos de perturbacion
   - Señales de control durante perturbaciones

3. **exp3_defuzzification.png**: Comparacion de metodos de defuzzificacion
   - Temperatura usando centroid, bisector y mean_of_maximum
   - Señales de control resultantes para cada metodo

### Metricas Mostradas

Para cada experimento se calcula:

- **Rise Time**: Tiempo de respuesta inicial
- **Overshoot**: Sobrepaso respecto al setpoint (%)
- **Settling Time**: Tiempo para estabilizarse
- **Steady-State Error**: Error en regimen permanente
- **IAE** (Integral Absolute Error): Area bajo la curva de error
- **ISE** (Integral Square Error): Area ponderada del error
- **ITAE** (Integral Time Absolute Error): Error ponderado por tiempo

## Detalles Tecnicos

### Controlador Fuzzy

- **Fuzzificacion**: Convierte valores crisp a grados de pertenencia
- **Inferencia**: Aplica 27 reglas Mamdani basadas en Temperatura y Error
- **Defuzzificacion**: Convierte salida difusa a crisp (3 metodos disponibles)

Variables difusas:
- Entrada: Temperatura (-5 a 50°C), Error (-10 a 10°C)
- Salida: Potencia (0 a 100%)
- Conjuntos: Negativo, Cero, Positivo con funciones de pertenencia triangulares

### Controlador PID

- Ecuacion: u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
- Anti-windup: Limita el termino integral
- Limites: Salida acotada entre 0% y 100%

### Modelo HVAC

Dinamica de primer orden:
```
dT/dt = -(T - T_ambient) / τ + gain * power
```

Parametros:
- τ (time_constant) = 5.0 minutos
- gain = 0.01 (relacion potencia-temperatura)

## Resultados Esperados

En condiciones normales (parametros por defecto):

- **PID**: Respuesta mas rapida, pero mayor overshoot
- **Fuzzy**: Respuesta mas suave, mejor manejo de regimen transitorio
- **Defuzzificacion**: Centroid y bisector dan resultados similares; mean_of_maximum produce respuesta ligeramente diferente

## Arquitectura

El codigo sigue principios de separacion de responsabilidades:

- `main.py`: Entrada de usuario y orquestacion (142 lineas)
- `src/experiments.py`: Logica de simulaciones y graficacion (310 lineas)
- `src/fuzzy_controller/`: Componentes del controlador difuso (5 modulos)
- `src/pid_controller/`: Implementacion del PID (1 modulo)
- `src/simulation/`: Modelo del sistema (1 modulo)

## Notas

- Los archivos Python han sido limpiados de caracteres especiales y comentarios extensos para optimizar legibilidad
- El directorio `results` se crea automaticamente al ejecutar el programa
- Se recomienda usar `test_input.txt` para reproducibilidad de experimentos
- Para cambiar parametros rapidamente, editar `test_input.txt` y redirigir


