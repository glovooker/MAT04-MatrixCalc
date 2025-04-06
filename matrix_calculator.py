from fractions import Fraction

# -------------------- MATRIX UTILS --------------------

def create_matrix(num_rows, num_cols, values):
    """Crea una matriz usando números exactos (Fraction)."""
    if len(values) != num_rows or any(len(row) != num_cols for row in values):
        raise ValueError("El tamaño de los datos no coincide con las dimensiones especificadas.")
    return [[Fraction(x) for x in row] for row in values]

def format_matrix(matrix):
    """Devuelve una cadena que muestra la matriz en un formato estético.
    Ejemplo:
    | 1 | 0 | 0 | 4 |
    | 0 | 1 | 0 | 5 |
    | 0 | 0 | 1 | 5 |"""
    formatted_rows = []
    for row in matrix:
        row_str = " | ".join(str(item) for item in row)
        formatted_rows.append(f"| {row_str} |")
    return "\n".join(formatted_rows)

def clone_matrix(matrix):
    """Devuelve una copia de la matriz."""
    return [list(row) for row in matrix]

def is_square(matrix):
    return len(matrix) == len(matrix[0])

# -------------------- GAUSS-JORDAN Y OPERACIONES RELACIONADAS --------------------

def gauss_jordan_elimination(matrix):
    """Realiza la eliminación Gauss-Jordan mostrando los pasos en español."""
    rows = len(matrix)
    cols = len(matrix[0])
    mat = clone_matrix(matrix)
    steps = []
    for i in range(rows):
        # Verificar pivote distinto de cero, intercambiar si es necesario
        if mat[i][i] == 0:
            swapped = False
            for j in range(i+1, rows):
                if mat[j][i] != 0:
                    mat[i], mat[j] = mat[j], mat[i]
                    steps.append(f"Se intercambia: fila {i+1} <-> fila {j+1}...")
                    swapped = True
                    break
            if not swapped:
                raise ValueError(f"No hay pivote no nulo en la fila {i+1}.")
        pivot = mat[i][i]
        mat[i] = [elem / pivot for elem in mat[i]]
        steps.append(f"Se normaliza la fila {i+1} al dividirla por {pivot}...")
        for j in range(rows):
            if j != i:
                factor = mat[j][i]
                mat[j] = [mat[j][k] - factor * mat[i][k] for k in range(cols)]
                steps.append(f"Se resta {factor} veces la fila {i+1} de la fila {j+1}...")
    print("\n".join(steps))
    return mat

def get_submatrix(matrix, exclude_row, exclude_col):
    """Devuelve la submatriz excluyendo la fila y columna indicadas."""
    return [
        [matrix[i][j] for j in range(len(matrix)) if j != exclude_col]
        for i in range(len(matrix)) if i != exclude_row
    ]

def determinant(matrix):
    """Calcula recursivamente el determinante de una matriz cuadrada."""
    if not is_square(matrix):
        raise ValueError("El determinante se calcula para matrices cuadradas.")
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    det = Fraction(0)
    for col in range(n):
        sign = Fraction((-1) ** col)
        sub = get_submatrix(matrix, 0, col)
        det += sign * matrix[0][col] * determinant(sub)
    return det

def adjugate(matrix):
    """Calcula la adjunta (matriz de cofactores) de una matriz cuadrada."""
    n = len(matrix)
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            sign = Fraction((-1) ** (i + j))
            sub = get_submatrix(matrix, i, j)
            row.append(sign * determinant(sub))
        adj.append(row)
    return adj

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def inverse_by_adjugate(matrix):
    """Calcula la inversa de una matriz usando el método de la adjunción."""
    if not is_square(matrix):
        raise ValueError("La matriz debe ser cuadrada.")
    det = determinant(matrix)
    if det == 0:
        raise ValueError("La matriz no se puede invertir porque su determinante es 0.")
    adj = adjugate(matrix)
    inv = transpose(adj)
    return [[elem / det for elem in row] for row in inv]

def inverse_by_gauss_jordan(matrix):
    """Calcula la inversa de una matriz mediante el método de Gauss-Jordan."""
    if not is_square(matrix):
        raise ValueError("La matriz debe ser cuadrada.")
    n = len(matrix)
    mat = clone_matrix(matrix)
    # Se crea la matriz aumentada
    augmented = [row + [Fraction(1) if i == j else Fraction(0) for j in range(n)] for i, row in enumerate(mat)]
    steps = []
    for i in range(n):
        if augmented[i][i] == 0:
            swapped = False
            for j in range(i+1, n):
                if augmented[j][i] != 0:
                    augmented[i], augmented[j] = augmented[j], augmented[i]
                    steps.append(f"Se intercambia: fila {i+1} <-> fila {j+1}...")
                    swapped = True
                    break
            if not swapped:
                raise ValueError(f"No hay pivote no nulo en la fila {i+1}.")
        pivot = augmented[i][i]
        augmented[i] = [elem / pivot for elem in augmented[i]]
        steps.append(f"Se divide la fila {i+1} por {pivot}...")
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - factor * augmented[i][k] for k in range(2 * n)]
                steps.append(f"Se resta {factor} veces la fila {i+1} a la fila {j+1}...")
    print("\n".join(steps))
    inverse_matrix = [row[n:] for row in augmented]
    return inverse_matrix

# -------------------- OPERACIONES BÁSICAS --------------------

def add_matrixes(m1, m2):
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Las matrices deben tener la misma cantidad de filas y columnas.")
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def subtract_matrixes(m1, m2):
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Las matrices deben tener la misma cantidad de filas y columnas.")
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def multiply_matrixes(m1, m2):
    if len(m1[0]) != len(m2):
        raise ValueError("El número de columnas de la primera matriz debe coincidir con el número de filas de la segunda")
    result = []
    for i in range(len(m1)):
        new_row = []
        for j in range(len(m2[0])):
            s = sum(m1[i][k] * m2[k][j] for k in range(len(m2)))
            new_row.append(s)
        result.append(new_row)
    return result

def scalar_multiply(matrix, scalar):
    return [[elem * scalar for elem in row] for row in matrix]

def solve_cramers_rule(aug_matrix):
    """Resuelve un sistema lineal usando la regla de Cramer.
    Se espera que la matriz aumentada tenga n filas y n+1 columnas."""
    n = len(aug_matrix)
    if any(len(row) != n + 1 for row in aug_matrix):
        raise ValueError("La matriz aumentada debe tener n filas y n+1 columnas")
    A = [row[:-1] for row in aug_matrix]
    b = [row[-1] for row in aug_matrix]
    det_A = determinant(A)
    if det_A == 0:
        raise ValueError("El sistema no tiene solución única (determinante de A es 0)")
    solution = []
    for i in range(n):
        Ai = clone_matrix(A)
        for r in range(n):
            Ai[r][i] = b[r]
        sol = determinant(Ai) / det_A
        solution.append(sol)
    return solution

# -------------------- MENÚ INTERACTIVO --------------------

def main_menu():
    matrixes = []  # Lista donde se almacenarán las matrices creadas
    while True:
        print("\n=======================================")
        print("BIENVENIDO A LA CALCULADORA DE MATRICES")
        print("=======================================")
        print("1. Agregar nueva matriz")
        print("2. Ver matrices agregadas")
        print("3. Realizar una operación")
        print("4. Salir")
        choice = input("Selecciona una opción: ").strip()
        
        if choice == "1":
            try:
                rows = int(input("Número de filas: "))
                cols = int(input("Número de columnas: "))
                data = []
                for i in range(rows):
                    row_input = input(f"Ingresa los {cols} valores de la fila {i+1} separados por espacios: ").split()
                    if len(row_input) != cols:
                        raise ValueError("La cantidad de valores no coincide con la cantidad de columnas")
                    data.append([float(x) for x in row_input])
                new_matrix = create_matrix(rows, cols, data)
                matrixes.append(new_matrix)
                print(f"Matriz agregada con ID {len(matrixes) - 1}")
            except Exception as e:
                print(f"Error al agregar la matriz: {e}")
        
        elif choice == "2":
            if not matrixes:
                print("No hay matrices agregadas aún.")
            else:
                print("Matrices agregadas:")
                for idx, mat in enumerate(matrixes):
                    print(f"\n[{idx}]:")
                    print(format_matrix(mat))
        
        elif choice == "3":
            if not matrixes:
                print("No existen matrices para operar. Agrega al menos una matriz.")
                continue
            while True:
                print("\n------------ OPERACIONES DISPONIBLES ------------")
                # Orden de opciones modificado para reflejar tu estilo personal
                print("1. Sumar dos matrices")
                print("2. Restar dos matrices")
                print("3. Multiplicar dos matrices")
                print("4. Multiplicación escalar")
                print("5. Eliminación Gauss-Jordan")
                print("6. Inversa (Gauss-Jordan)")
                print("7. Inversa (Adjunción)")
                print("8. Determinante")
                print("9. Regla de Cramer")
                print("10. Volver al menú principal")
                oper = input("Elige una operación: ").strip().lower()
                
                if oper == "10":
                    break
                
                try:
                    if oper in ("1", "2", "3"):
                        idx1 = int(input("ID de la primera matriz: "))
                        idx2 = int(input("ID de la segunda matriz: "))
                        if oper == "1":
                            result = add_matrixes(matrixes[idx1], matrixes[idx2])
                        elif oper == "2":
                            result = subtract_matrixes(matrixes[idx1], matrixes[idx2])
                        else:  # oper == "3"
                            result = multiply_matrixes(matrixes[idx1], matrixes[idx2])
                    
                    elif oper == "4":
                        idx = int(input("ID de la matriz: "))
                        scalar = float(input("Ingrese el escalar: "))
                        result = scalar_multiply(matrixes[idx], scalar)
                    
                    elif oper == "5":
                        idx = int(input("ID de la matriz: "))
                        result = gauss_jordan_elimination(matrixes[idx])
                    
                    elif oper == "6":
                        idx = int(input("ID de la matriz: "))
                        result = inverse_by_gauss_jordan(matrixes[idx])
                    
                    elif oper == "7":
                        idx = int(input("ID de la matriz: "))
                        result = inverse_by_adjugate(matrixes[idx])
                    
                    elif oper == "8":
                        idx = int(input("ID de la matriz: "))
                        det = determinant(matrixes[idx])
                        print(f"Determinante: {det}")
                        continue
                    
                    elif oper == "9":
                        idx = int(input("ID de la matriz: "))
                        sol = solve_cramers_rule(matrixes[idx])
                        print("Resultado:")
                        for s in sol:
                            print(s)
                        continue
                    
                    else:
                        print("Opción inválida. Intenta de nuevo.")
                        continue
                    
                    # Mostrar resultado de la operación con formato especial
                    print("\nResultado de la operación:")
                    if isinstance(result, list):
                        print(format_matrix(result))
                    else:
                        print(result)
                except Exception as e:
                    print(f"Error durante la operación: {e}")
        
        elif choice == "4":
            print("Saliendo... ¡Hasta la próxima!")
            break
        else:
            print("Opción inválida. Por favor, intenta de nuevo.")

if __name__ == "__main__":
    main_menu()
