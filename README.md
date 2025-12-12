# Método de Newton-Raphson para sistemas no lineales

Aplicación interactiva de escritorio escrita en Python/Tkinter que resuelve sistemas de ecuaciones no lineales mediante el método de Newton-Raphson. Calcula automáticamente el Jacobiano usando derivación simbólica y muestra cada iteración en una tabla interactiva con métricas de convergencia.

## Características principales

✓ **Sistemas de cualquier tamaño**: Resuelve sistemas con n ecuaciones y n variables  
✓ **Jacobiano automático**: Derivación simbólica usando SymPy (sin necesidad de derivadas manuales)  
✓ **Tabla interactiva**: Visualiza todos los pasos del método con:

- Valores de variables en cada iteración
- Norma infinito del cambio: $\lVert \Delta x \rVert_\infty$
- Norma infinito del residuo: $\lVert F(x) \rVert_\infty$

✓ **Parámetros configurables**: Tolerancia, máximo de iteraciones, vector inicial  
✓ **Validación robusta**: Detección de problemas (Jacobiano singular, entrada inválida, no convergencia)  
✓ **Funciones matemáticas incluidas**: Trigonométricas, exponenciales, logarítmicas, etc.

## Requisitos del sistema

- **Python**: 3.10 o superior (probado con 3.13)
- **Dependencias**: NumPy, SymPy, Tkinter (incluido en Python por defecto)

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Guía de uso

### 1. Ejecutar la aplicación

```bash
python newton_gui.py
```

### 2. Completar el formulario

| Campo                    | Ejemplo                       | Descripción                                             |
| ------------------------ | ----------------------------- | ------------------------------------------------------- |
| **Variables**            | `x, y`                        | Nombres separados por coma                              |
| **Ecuaciones**           | `x**2 + y**2 - 1`<br/>`x - y` | Una por línea; iguales a cero implícitamente            |
| **Aproximación inicial** | `0.5, 0.5`                    | Punto de partida para iteraciones                       |
| **Tolerancia**           | `1e-6`                        | Criterio de parada para $\lVert \Delta x \rVert_\infty$ |
| **Iteraciones máximas**  | `20`                          | Límite de iteraciones                                   |

### 3. Ver resultados

- **Tabla de iteraciones**: Cada paso del algoritmo con métricas
- **Solución final**: Resumen de convergencia
- **Botón Limpiar**: Restaura valores por defecto

## Funciones disponibles

**Trigonométricas**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`  
**Hiperbólicas**: `sinh`, `cosh`, `tanh`  
**Especiales**: `exp`, `log`, `sqrt`, `pi`, `e`

## Resolución de problemas

| Problema             | Causa                              | Solución                      |
| -------------------- | ---------------------------------- | ----------------------------- |
| "Jacobiana singular" | Aproximación inicial inadecuada    | Intenta otro punto de partida |
| "No converge"        | Límite de iteraciones insuficiente | Aumenta máximo de iteraciones |
| Error de sintaxis    | Ecuación inválida (ej. `x^2`)      | Usa sintaxis Python: `x**2`   |

## Ejemplo práctico

**Sistema**:

- $x^2 + y^2 = 1$
- $x - y = 0$

**Entrada**:

```
Variables: x, y
Ecuaciones: x**2 + y**2 - 1
            x - y
Inicial: 0.5, 0.5
```

**Resultado esperado**: $x \approx 0.707, y \approx 0.707$ en ~3 iteraciones

## Archivos

- `newton_gui.py` - Aplicación principal
- `requirements.txt` - Dependencias
- `main.tex` - Documentación técnica
