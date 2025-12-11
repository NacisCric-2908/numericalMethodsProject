# numericalMethodsProject

Aplicación educativa en Python para resolver sistemas de ecuaciones no lineales
mediante el método de Newton-Raphson. Permite resolver sistemas 2x2 y 3x3 con
interfaz gráfica (Tkinter), muestra el historial de iteraciones y verifica las
funciones en la solución encontrada.

## Contenido

- `trabajo.py` — Aplicación principal: interfaz Tkinter y la implementación del
	método de Newton-Raphson para sistemas 2x2 y 3x3.

## Características

- Interfaz gráfica simple construida con Tkinter.
- Soporta sistemas 2x2 y 3x3.
- Entrada de funciones en sintaxis Python (usar `x`, `y`, `z` y `**` para potencias).
- Muestra historial de iteraciones y verificación de las funciones en la solución.
- La interfaz ahora es desplazable (scroll vertical) para pantallas pequeñas.

## Requisitos

- Python 3.8+ (probado con Python 3.10)
- Paquetes de Python:
	- numpy
	- sympy

Tkinter suele venir incluido con la mayoría de distribuciones de Python; si no
está presente, instálalo según tu distribución (por ejemplo, `sudo apt install
python3-tk` en Debian/Ubuntu).

Instala dependencias con pip si es necesario:

```bash
python3 -m pip install --user numpy sympy
```

## Uso

1. Abrir una terminal en la carpeta del proyecto.
2. Ejecutar la aplicación:

```bash
python3 trabajo.py
```

Se abrirá una ventana con la interfaz. Pasos básicos:

- Selecciona si quieres resolver un sistema `2x2` o `3x3`.
- Escribe las ecuaciones en las entradas correspondientes usando `x, y (z)`.
	- Ejemplos: `x**2 + y**2 - 4`, `x*y - 1`, `x + y - z**2`.
- Introduce los valores iniciales x₀, y₀ (y z₀ si aplica).
- Ajusta la tolerancia y el número máximo de iteraciones si lo deseas.
- Pulsa `Resolver Sistema` para ejecutar el algoritmo. Los resultados y el
	historial aparecerán en el panel de resultados.

### Comportamiento en pantallas pequeñas

La interfaz está ahora dentro de un `Canvas` con una barra de desplazamiento
vertical. Si no puedes ver todo el contenido en una pantalla pequeña, usa la
barra de la derecha o la rueda del ratón para desplazarte verticalmente. Si la
rueda no responde en tu entorno, prueba usar la barra o indícame tu entorno de
escritorio y lo ajusto.

## Notas de implementación

- El archivo `trabajo.py` contiene las funciones clave:
	- `parse_equation(eq_str, n_vars)`: convierte la cadena a una expresión SymPy
		y valida variables permitidas.
	- `newton_raphson(equations, initial_guess, tol, max_iter)`: implementa el
		método de Newton-Raphson para sistemas.
	- `NewtonRaphsonApp`: clase que construye la UI y conecta la lógica.

- Seguridad: el parseo de ecuaciones descarta palabras potencialmente
	peligrosas como `import`, `exec`, `eval`, `__`, `open` y `file`.

## Estructura de archivos

```
numericalMethodsProject/
├─ trabajo.py         # Aplicación principal
├─ README.md          # Documentación (este archivo)
└─ __pycache__/       # Archivos compilados de Python
```

## Pruebas rápidas

- Ejecuta `python3 trabajo.py` y prueba resolver el sistema de ejemplo 3x3
	(las ecuaciones cargadas por defecto) con los valores iniciales provistos.

## Posibles mejoras

- Añadir pruebas unitarias para la función `newton_raphson` (por ejemplo,
	casos simples con soluciones conocidas).
- Implementar guardado/carga de sistemas (ficheros JSON) para reproducir casos.
- Mejorar la validación y manejo de errores (por ejemplo, jacobiano cercano a
	singularidad).

## Autor

Proyecto creado por NacisCric-2908.

---
