"""GUI para resolver sistemas de ecuaciones no lineales con Newton-Raphson."""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Sequence, Tuple

import numpy as np
import sympy as sp


@dataclass
class IterationRecord:
    index: int
    values: List[float]
    delta_norm: float
    residual_norm: float


class NewtonSystemSolver:
    """Resuelve sistemas no lineales usando Newton-Raphson."""

    def __init__(self) -> None:
        self._safe_locals = self._build_safe_locals()

    @staticmethod
    def _build_safe_locals() -> Dict[str, object]:
        allowed = [
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "exp",
            "log",
            "sqrt",
            "pi",
            "E",
        ]
        safe = {name: getattr(sp, name) for name in allowed if hasattr(sp, name)}
        safe.update({"pi": sp.pi, "e": sp.E})
        return safe

    @staticmethod
    def _parse_variables(raw: str) -> List[str]:
        variables = [token.strip() for token in raw.split(",") if token.strip()]
        if not variables:
            raise ValueError("Debe ingresar al menos una variable, separada por comas.")
        if len(set(variables)) != len(variables):
            raise ValueError("Las variables no pueden repetirse.")
        return variables

    @staticmethod
    def _parse_initial_guess(raw: str, expected: int) -> np.ndarray:
        try:
            values = [float(token.strip()) for token in raw.split(",") if token.strip()]
        except ValueError as exc:
            raise ValueError("La aproximación inicial debe contener solo números.") from exc
        if len(values) != expected:
            raise ValueError(
                f"La aproximación inicial debe tener {expected} valores; se recibieron {len(values)}."
            )
        return np.array(values, dtype=float)

    def _parse_functions(self, raw: str, variables: Sequence[str]) -> Tuple[List[sp.Expr], sp.MutableDenseMatrix]:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Debe ingresar al menos una ecuación.")
        if len(lines) != len(variables):
            raise ValueError(
                f"El número de ecuaciones ({len(lines)}) debe coincidir con el número de variables ({len(variables)})."
            )
        symbols = sp.symbols(variables)
        expressions = []
        for line in lines:
            try:
                expr = sp.sympify(line, locals=self._safe_locals)
            except (sp.SympifyError, TypeError) as exc:
                raise ValueError(f"No se pudo interpretar la expresión: '{line}'.") from exc
            expressions.append(expr)
        jacobian = sp.Matrix(expressions).jacobian(symbols)
        return expressions, jacobian

    def solve(
        self,
        functions_raw: str,
        variables_raw: str,
        initial_guess_raw: str,
        tolerance: float,
        max_iterations: int,
    ) -> Tuple[List[IterationRecord], np.ndarray, bool]:
        variables = self._parse_variables(variables_raw)
        initial_guess = self._parse_initial_guess(initial_guess_raw, len(variables))
        expressions, jacobian = self._parse_functions(functions_raw, variables)

        symbols = sp.symbols(variables)
        f_lambda = sp.lambdify(symbols, expressions, modules="numpy")
        j_lambda = sp.lambdify(symbols, jacobian, modules="numpy")

        current = initial_guess.astype(float)
        iterations: List[IterationRecord] = []

        for idx in range(1, max_iterations + 1):
            f_val = np.array(f_lambda(*current), dtype=float).reshape(-1)
            j_val = np.array(j_lambda(*current), dtype=float)

            try:
                delta = np.linalg.solve(j_val, -f_val)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "La matriz Jacobiana es singular o está mal condicionada en esta iteración." 
                    " Intente otra aproximación inicial."
                ) from exc

            next_guess = current + delta
            delta_norm = float(np.linalg.norm(delta, ord=np.inf))
            residual_norm = float(np.linalg.norm(f_val, ord=np.inf))

            iterations.append(
                IterationRecord(
                    index=idx,
                    values=next_guess.tolist(),
                    delta_norm=delta_norm,
                    residual_norm=residual_norm,
                )
            )

            current = next_guess
            if delta_norm < tolerance:
                return iterations, current, True

        return iterations, current, False


class NewtonGUIApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Newton-Raphson para sistemas no lineales")
        self.solver = NewtonSystemSolver()
        self.result_var = tk.StringVar(value="Ingrese datos y presione 'Resolver'.")
        self._build_ui()

    def _build_ui(self) -> None:
        self.root.geometry("900x600")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        input_frame = ttk.LabelFrame(self.root, text="Datos de entrada")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(input_frame, text="Variables (separadas por coma):").grid(row=0, column=0, sticky="w")
        self.variables_entry = ttk.Entry(input_frame)
        self.variables_entry.insert(0, "x, y")
        self.variables_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(input_frame, text="Ecuaciones (una por línea):").grid(row=1, column=0, sticky="nw")
        self.functions_text = ScrolledText(input_frame, height=5, width=40)
        self.functions_text.insert("1.0", "x**2 + y**2 - 1\nx - y")
        self.functions_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(input_frame, text="Aproximación inicial (coma):").grid(row=2, column=0, sticky="w")
        self.initial_entry = ttk.Entry(input_frame)
        self.initial_entry.insert(0, "0.5, 0.5")
        self.initial_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(input_frame, text="Tolerancia:").grid(row=3, column=0, sticky="w")
        self.tolerance_entry = ttk.Entry(input_frame)
        self.tolerance_entry.insert(0, "1e-6")
        self.tolerance_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(input_frame, text="Iteraciones máximas:").grid(row=4, column=0, sticky="w")
        self.iterations_entry = ttk.Entry(input_frame)
        self.iterations_entry.insert(0, "20")
        self.iterations_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")
        button_frame.grid_columnconfigure((0, 1), weight=1)

        solve_btn = ttk.Button(button_frame, text="Resolver", command=self._on_solve)
        solve_btn.grid(row=0, column=0, padx=5, sticky="ew")

        clear_btn = ttk.Button(button_frame, text="Limpiar", command=self._clear_fields)
        clear_btn.grid(row=0, column=1, padx=5, sticky="ew")

        # Tabla de iteraciones
        table_frame = ttk.LabelFrame(self.root, text="Iteraciones")
        table_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, columns=(), show="headings", height=10)
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        ttk.Label(self.root, textvariable=self.result_var, font=("Helvetica", 11, "bold"), wraplength=850).grid(
            row=2, column=0, padx=10, pady=10, sticky="w"
        )

    def _configure_tree(self, variables: Sequence[str]) -> None:
        columns = ["Iteración", *variables, "||Δx||∞", "||F(x)||∞"]
        self.tree.configure(columns=columns)
        for col in columns:
            self.tree.heading(col, text=col)
            anchor = "center" if col == "Iteración" else "e"
            width = 100 if col != "Iteración" else 80
            self.tree.column(col, anchor=anchor, width=width, stretch=True)

    def _clear_table(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)

    def _clear_fields(self) -> None:
        self.variables_entry.delete(0, tk.END)
        self.variables_entry.insert(0, "x, y")
        self.functions_text.delete("1.0", tk.END)
        self.functions_text.insert("1.0", "x**2 + y**2 - 1\nx - y")
        self.initial_entry.delete(0, tk.END)
        self.initial_entry.insert(0, "0.5, 0.5")
        self.tolerance_entry.delete(0, tk.END)
        self.tolerance_entry.insert(0, "1e-6")
        self.iterations_entry.delete(0, tk.END)
        self.iterations_entry.insert(0, "20")
        self._clear_table()
        self.result_var.set("Campos restablecidos. Ingrese nuevos valores.")

    def _on_solve(self) -> None:
        functions_raw = self.functions_text.get("1.0", tk.END)
        variables_raw = self.variables_entry.get()
        initial_raw = self.initial_entry.get()

        try:
            tolerance = float(self.tolerance_entry.get())
            max_iter = int(self.iterations_entry.get())
            if tolerance <= 0 or max_iter <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "La tolerancia debe ser positiva y las iteraciones un entero positivo.")
            return

        try:
            iterations, final_vector, converged = self.solver.solve(
                functions_raw=functions_raw,
                variables_raw=variables_raw,
                initial_guess_raw=initial_raw,
                tolerance=tolerance,
                max_iterations=max_iter,
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        variables = [token.strip() for token in variables_raw.split(",") if token.strip()]
        self._configure_tree(variables)
        self._clear_table()

        for record in iterations:
            row = [record.index]
            row.extend(f"{val:.8f}" for val in record.values)
            row.append(f"{record.delta_norm:.3e}")
            row.append(f"{record.residual_norm:.3e}")
            self.tree.insert("", tk.END, values=row)

        formatted_solution = ", ".join(f"{var}={val:.8f}" for var, val in zip(variables, final_vector))
        if converged:
            self.result_var.set(
                f"Convergencia alcanzada en {len(iterations)} iteraciones. Solución aproximada: {formatted_solution}"
            )
        else:
            self.result_var.set(
                f"No se alcanzó la tolerancia tras {len(iterations)} iteraciones. Última aproximación: {formatted_solution}"
            )

    @staticmethod
    def run() -> None:
        root = tk.Tk()
        app = NewtonGUIApp(root)
        root.mainloop()


if __name__ == "__main__":
    NewtonGUIApp.run()
