"""
Sistema de Ecuaciones No Lineales - Método Newton-Raphson
Permite resolver sistemas 2x2 y 3x3
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import sympy as sp
from sympy import symbols, lambdify


class NewtonRaphsonApp:
    """Aplicación para resolver sistemas de ecuaciones no lineales"""

    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Ecuaciones No Lineales - Newton-Raphson")
        self.root.geometry("950x800")

        # Variables simbólicas
        self.x, self.y, self.z = symbols('x y z')
        self.system_size = tk.StringVar(value="3x3")

        # Widgets que se crean dinámicamente
        self.eq_labels = []
        self.eq_entries = []
        self.initial_labels = []
        self.initial_entries = []

        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Crear un contenedor con Canvas y scrollbar para permitir scroll vertical
        # en pantallas pequeñas o ventanas reducidas
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Asegurar que la raíz y el contenedor puedan expandirse
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(container)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.grid(row=0, column=1, sticky='ns')
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Frame que contendrá todos los widgets de la UI dentro del canvas
        main_frame = ttk.Frame(canvas, padding="10")
        canvas.create_window((0, 0), window=main_frame, anchor='nw')

        # Actualizar la región de scroll cuando cambie el tamaño del frame
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        main_frame.bind("<Configure>", _on_frame_configure)

        # Soporte para la rueda del ratón (Windows/Mac/Linux)
        def _on_mousewheel(event):
            # event.delta (Windows/Mac), event.num (Linux Button-4/5)
            try:
                if event.num == 5 or event.delta < 0:
                    canvas.yview_scroll(1, "units")
                elif event.num == 4 or event.delta > 0:
                    canvas.yview_scroll(-1, "units")
            except Exception:
                # Fallback genérico
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Bind de rueda para varias plataformas
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        # Título
        title_label = ttk.Label(
            main_frame,
            text="Sistema de Ecuaciones No Lineales - Newton-Raphson",
            font=('Arial', 14, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Selección de tamaño del sistema
        size_frame = ttk.LabelFrame(main_frame, text="Tamaño del Sistema", padding="10")
        size_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))

        ttk.Radiobutton(
            size_frame, text="2x2 (dos ecuaciones, dos incógnitas)",
            variable=self.system_size, value="2x2", command=self.update_system_size
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            size_frame, text="3x3 (tres ecuaciones, tres incógnitas)",
            variable=self.system_size, value="3x3", command=self.update_system_size
        ).pack(anchor=tk.W)

        # Instrucciones
        instructions_text = """Instrucciones:
• Use x, y (y z para 3x3) como variables
• Sintaxis Python: use ** para potencias, * para multiplicaciones
• Funciones: sin, cos, tan, exp, log, sqrt, etc.
• Ejemplos: x**2 + y - 1, sin(x) + cos(y) - 0.5"""

        instructions = ttk.Label(main_frame, text=instructions_text, justify=tk.LEFT)
        instructions.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        # Frame para ecuaciones (se llena dinámicamente)
        self.equations_frame = ttk.LabelFrame(
            main_frame, text="Ecuaciones del Sistema", padding="10"
        )
        self.equations_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10),
                                  sticky=(tk.W, tk.E))

        # Frame para valores iniciales (se llena dinámicamente)
        self.initials_frame = ttk.LabelFrame(
            main_frame, text="Valores Iniciales", padding="10"
        )
        self.initials_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10),
                                sticky=(tk.W, tk.E))

        # Parámetros del método
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros", padding="10")
        params_frame.grid(row=5, column=0, columnspan=2, pady=(0, 10),
                         sticky=(tk.W, tk.E))

        ttk.Label(params_frame, text="Tolerancia:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.tol_entry = ttk.Entry(params_frame, width=20)
        self.tol_entry.grid(row=0, column=1, pady=5, padx=(10, 0))
        self.tol_entry.insert(0, "1e-6")

        ttk.Label(params_frame, text="Iteraciones máximas:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.max_iter_entry = ttk.Entry(params_frame, width=20)
        self.max_iter_entry.grid(row=1, column=1, pady=5, padx=(10, 0))
        self.max_iter_entry.insert(0, "100")

        # Botón resolver
        solve_button = ttk.Button(
            main_frame, text="Resolver Sistema", command=self.solve_system
        )
        solve_button.grid(row=6, column=0, columnspan=2, pady=(0, 10))

        # Área de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        results_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.results_text = scrolledtext.ScrolledText(
            results_frame, width=100, height=15, wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Configurar grid para que se expanda
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)

        # Inicializar con sistema 3x3
        self.update_system_size()

    def update_system_size(self):
        """Actualiza la interfaz según el tamaño del sistema seleccionado"""
        # Limpiar frames
        for widget in self.equations_frame.winfo_children():
            widget.destroy()
        for widget in self.initials_frame.winfo_children():
            widget.destroy()

        self.eq_labels.clear()
        self.eq_entries.clear()
        self.initial_labels.clear()
        self.initial_entries.clear()

        size = self.system_size.get()
        n_equations = 2 if size == "2x2" else 3

        # Crear entradas para ecuaciones
        equations_examples = [
            "x**2 + y**2 - 4",
            "x*y - 1",
            "x + y - z**2"
        ]

        variables_text = ["x, y", "x, y", "x, y, z"]

        for i in range(n_equations):
            label = ttk.Label(
                self.equations_frame,
                text=f"Ecuación {i+1}: f{i+1}({variables_text[min(i, 2)]}) = 0"
            )
            label.grid(row=i, column=0, sticky=tk.W, pady=5)
            self.eq_labels.append(label)

            entry = ttk.Entry(self.equations_frame, width=60)
            entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            entry.insert(0, equations_examples[i])
            self.eq_entries.append(entry)

        # Crear entradas para valores iniciales
        var_names = ['x', 'y', 'z']
        initial_values = ['1.5', '1.5', '1.5']

        for i in range(n_equations):
            label = ttk.Label(self.initials_frame, text=f"{var_names[i]}₀:")
            label.grid(row=i, column=0, sticky=tk.W, pady=5)
            self.initial_labels.append(label)

            entry = ttk.Entry(self.initials_frame, width=20)
            entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            entry.insert(0, initial_values[i])
            self.initial_entries.append(entry)

    def parse_equation(self, eq_str, n_vars):
        """Convierte string de ecuación a expresión sympy"""
        try:
            eq_str = eq_str.strip().replace('^', '**')

            # Verificar caracteres peligrosos
            dangerous = ['import', 'exec', 'eval', '__', 'open', 'file']
            if any(word in eq_str.lower() for word in dangerous):
                raise ValueError("Expresión contiene palabras no permitidas")

            # Convertir a expresión sympy
            expr = sp.sympify(eq_str)

            # Verificar variables permitidas
            allowed_vars = [self.x, self.y] if n_vars == 2 else [self.x, self.y, self.z]
            if not expr.free_symbols.issubset(set(allowed_vars)):
                invalid = expr.free_symbols - set(allowed_vars)
                raise ValueError(f"Variables no permitidas: {invalid}")

            return expr

        except sp.SympifyError as e:
            raise ValueError(f"Error de sintaxis: {eq_str}") from e
        except Exception as e:
            raise ValueError(f"Error en la ecuación: {str(e)}") from e

    def newton_raphson(self, equations, initial_guess, tol, max_iter):
        """
        Implementa el método de Newton-Raphson para sistemas no lineales

        Args:
            equations: lista de expresiones sympy
            initial_guess: vector con valores iniciales
            tol: tolerancia para convergencia
            max_iter: número máximo de iteraciones

        Returns:
            tuple: (solución, iteraciones realizadas, convergió)
        """
        n = len(equations)
        vars_list = [self.x, self.y] if n == 2 else [self.x, self.y, self.z]

        # Calcular matriz Jacobiana simbólicamente
        jacobian_matrix = sp.Matrix(
            [[sp.diff(eq, var) for var in vars_list] for eq in equations]
        )

        # Convertir a funciones numéricas
        f_funcs = [lambdify(vars_list, eq, modules=['numpy']) for eq in equations]
        j_funcs = [[lambdify(vars_list, jacobian_matrix[i, j], modules=['numpy'])
                   for j in range(n)] for i in range(n)]

        # Iteración de Newton-Raphson
        x_current = np.array(initial_guess, dtype=float)
        iteration_history = [x_current.copy()]

        for iteration in range(max_iter):
            # Evaluar F(x)
            f_values = np.array([f_funcs[i](*x_current) for i in range(n)])

            # Evaluar J(x)
            j_values = np.array(
                [[j_funcs[i][j](*x_current) for j in range(n)] for i in range(n)]
            )

            # Verificar si la matriz jacobiana es singular
            if np.linalg.det(j_values) == 0:
                return x_current, iteration + 1, False, iteration_history

            # Resolver J(x) * delta_x = -F(x)
            delta_x = np.linalg.solve(j_values, -f_values)

            # Actualizar x
            x_current = x_current + delta_x
            iteration_history.append(x_current.copy())

            # Verificar convergencia
            if np.linalg.norm(delta_x) < tol:
                return x_current, iteration + 1, True, iteration_history

        return x_current, max_iter, False, iteration_history

    def solve_system(self):
        """Resuelve el sistema de ecuaciones"""
        try:
            size = self.system_size.get()
            n = 2 if size == "2x2" else 3

            # Obtener ecuaciones
            eq_strings = [entry.get().strip() for entry in self.eq_entries]

            if not all(eq_strings):
                messagebox.showerror("Error", "Por favor ingrese todas las ecuaciones.")
                return

            # Parsear ecuaciones
            equations = [self.parse_equation(eq, n) for eq in eq_strings]

            # Obtener valores iniciales
            initial_guess = [float(entry.get()) for entry in self.initial_entries]

            # Obtener parámetros
            tol = float(self.tol_entry.get())
            max_iter = int(self.max_iter_entry.get())

            # Mostrar proceso
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Resolviendo sistema {size}...\n\n")
            self.results_text.insert(tk.END, "Ecuaciones:\n")
            for i, eq in enumerate(eq_strings):
                self.results_text.insert(tk.END, f"  f{i+1}: {eq} = 0\n")
            self.results_text.insert(tk.END, f"\nValores iniciales: {initial_guess}\n")
            self.results_text.insert(tk.END, f"Tolerancia: {tol}\n")
            self.results_text.insert(tk.END, f"Iteraciones máximas: {max_iter}\n\n")
            self.root.update()

            # Resolver
            solution, iterations, converged, history = self.newton_raphson(
                equations, initial_guess, tol, max_iter
            )

            # Mostrar resultados
            self.results_text.insert(tk.END, "="*70 + "\n")
            self.results_text.insert(tk.END, "RESULTADOS\n")
            self.results_text.insert(tk.END, "="*70 + "\n\n")

            if converged:
                self.results_text.insert(
                    tk.END, f"✓ Convergencia alcanzada en {iterations} iteraciones\n\n"
                )
            else:
                self.results_text.insert(
                    tk.END,
                    f"✗ No convergió en {iterations} iteraciones "
                    f"(puede ser una solución aproximada)\n\n"
                )

            # Solución
            var_names = ['x', 'y', 'z']
            self.results_text.insert(tk.END, "Solución encontrada:\n")
            for i in range(n):
                self.results_text.insert(
                    tk.END, f"  {var_names[i]} = {solution[i]:.10f}\n"
                )

            # Verificación
            self.results_text.insert(tk.END, "\nVerificación (valores de las funciones):\n")
            vars_list = [self.x, self.y] if n == 2 else [self.x, self.y, self.z]
            f_funcs = [lambdify(vars_list, eq, modules=['numpy']) for eq in equations]

            for i in range(n):
                value = f_funcs[i](*solution)
                self.results_text.insert(tk.END, f"  f{i+1}(solución) = {value:.2e}\n")

            # Historial de iteraciones
            self.results_text.insert(tk.END, "\nHistorial de iteraciones:\n")
            self.results_text.insert(
                tk.END, f"{'Iter':<6} {' '.join([f'{var_names[i]:>15}' for i in range(n)])}\n"
            )
            self.results_text.insert(tk.END, "-"*70 + "\n")

            for i, values in enumerate(history):
                values_str = ' '.join([f'{values[j]:>15.8f}' for j in range(n)])
                self.results_text.insert(tk.END, f"{i:<6} {values_str}\n")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado:\n{str(e)}")


def main():
    """Función principal"""
    root = tk.Tk()
    NewtonRaphsonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
