import numpy as np
import sympy as sp

def newton_raphson_sistema(functions, variables, x0, tol=1e-6, max_iter=100):
    """
    Método de Newton-Raphson para sistemas de ecuaciones no lineales
    
    Parameters:
    functions: lista de funciones [f1, f2, ...]
    variables: lista de variables [x1, x2, ...]
    x0: vector inicial
    tol: tolerancia
    max_iter: número máximo de iteraciones
    
    Returns:
    solución aproximada y número de iteraciones
    """
    
    n = len(functions)
    x = sp.Matrix(variables)
    f = sp.Matrix(functions)
    
    # Calcular la matriz jacobiana
    J = f.jacobian(x)
    
    # Convertir a funciones numéricas
    f_num = sp.lambdify([x], f, 'numpy')
    J_num = sp.lambdify([x], J, 'numpy')
    
    x_k = np.array(x0, dtype=float)
    
    print("Iteración\t", end="")
    for var in variables:
        print(f"{var}\t\t", end="")
    print("Error")
    print("-" * 60)
    
    for k in range(max_iter):
        f_val = f_num(x_k).flatten()
        J_val = J_num(x_k)
        
        # Resolver J * Δx = -f
        try:
            delta_x = np.linalg.solve(J_val, -f_val)
        except np.linalg.LinAlgError:
            print("Error: Matriz jacobiana singular")
            return None, k
        
        x_k1 = x_k + delta_x
        error = np.linalg.norm(delta_x)
        
        print(f"{k+1}\t\t", end="")
        for val in x_k1:
            print(f"{val:.6f}\t", end="")
        print(f"{error:.2e}")
        
        if error < tol:
            print(f"\nConvergencia alcanzada en {k+1} iteraciones")
            return x_k1, k+1
        
        x_k = x_k1
    
    print(f"\nMáximo número de iteraciones alcanzado")
    return x_k, max_iter

# Ejemplo 1: Sistema 2x2
def ejemplo_1():
    print("=== EJEMPLO 1: Sistema 2x2 ===")
    print("Ecuaciones:")
    print("f1(x,y) = x² + y² - 4 = 0")
    print("f2(x,y) = e^x + y - 1 = 0")
    print()
    
    x, y = sp.symbols('x y')
    f1 = x**2 + y**2 - 4
    f2 = sp.exp(x) + y - 1
    
    # Punto inicial
    x0 = [1.0, 1.0]
    
    solucion, iteraciones = newton_raphson_sistema(
        [f1, f2], [x, y], x0, tol=1e-8
    )
    
    if solucion is not None:
        print(f"\nSolución aproximada: x = {solucion[0]:.8f}, y = {solucion[1]:.8f}")
        
        # Verificación
        x_sol, y_sol = solucion
        error1 = x_sol**2 + y_sol**2 - 4
        error2 = np.exp(x_sol) + y_sol - 1
        print(f"Verificación: f1(x,y) = {error1:.2e}, f2(x,y) = {error2:.2e}")

# Ejemplo 2: Sistema 3x3
def ejemplo_2():
    print("\n=== EJEMPLO 2: Sistema 3x3 ===")
    print("Ecuaciones:")
    print("f1(x,y,z) = x + y + z - 3 = 0")
    print("f2(x,y,z) = x² + y² + z² - 5 = 0")
    print("f3(x,y,z) = e^x + x*y - x*z - 1 = 0")
    print()
    
    x, y, z = sp.symbols('x y z')
    f1 = x + y + z - 3
    f2 = x**2 + y**2 + z**2 - 5
    f3 = sp.exp(x) + x*y - x*z - 1
    
    x0 = [0.5, 1.0, 1.5]
    
    solucion, iteraciones = newton_raphson_sistema(
        [f1, f2, f3], [x, y, z], x0, tol=1e-8
    )
    
    if solucion is not None:
        print(f"\nSolución: x = {solucion[0]:.8f}, y = {solucion[1]:.8f}, z = {solucion[2]:.8f}")

if __name__ == "__main__":
    ejemplo_1()
    ejemplo_2()