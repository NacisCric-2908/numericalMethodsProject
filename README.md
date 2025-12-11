# Método de Newton-Raphson para sistemas no lineales

Aplicación de escritorio escrita en Python/Tkinter que resuelve sistemas de ecuaciones no lineales mediante el método de Newton-Raphson. Cada iteración se muestra en una tabla para que puedas seguir el progreso de la aproximación, junto con la última solución encontrada.

## Características

- Soporta sistemas de cualquier tamaño (mismo número de ecuaciones y variables).
- Deriva automáticamente el Jacobiano usando *SymPy*.
- Tabla interactiva con cada iteración: valores de las variables, norma del incremento $\lVert \Delta x \rVert_\infty$ y norma del residuo $\lVert F(x) \rVert_\infty$.
- Permite configurar tolerancia, máximo de iteraciones y vector inicial.
- Mensajes claros cuando el método converge o falla (Jacobian singular, datos inválidos, etc.).

## Requisitos

- Python 3.10 o superior (probado con 3.13)
- Dependencias listadas en `requirements.txt`

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecuta la aplicación:

```bash
python newton_gui.py
```

2. Completa los campos del formulario:
   - **Variables**: nombres separados por comas (ej. `x, y`).
   - **Ecuaciones**: una por línea, usando la sintaxis de Python/SymPy (ej. `x**2 + y**2 - 1`).
   - **Aproximación inicial**: valores separados por comas.
   - **Tolerancia** y **iteraciones** máximas.

3. Presiona **Resolver** para iniciar el método. Cada iteración aparecerá en la tabla inferior.
4. Usa **Limpiar** para restablecer los valores de ejemplo.

## Notas

- Las ecuaciones deben coincidir en número con las variables.
- Si el Jacobiano se vuelve singular, prueba con una aproximación inicial distinta.
- Puedes emplear funciones comunes (`sin`, `cos`, `exp`, `log`, etc.) y constantes (`pi`, `e`).
