from matrix import MMatrix
from typing import List


def center_data(X: MMatrix) -> MMatrix:
    n = X.n
    m = X.m
    result = MMatrix(n, m)
    means = []
    for j in range(m):
        total = 0.0
        for i in range(n):
            total += X.get(i + 1, j + 1)
        means.append(total / n)
    for i in range(n):
        for j in range(m):
            result.set(i + 1, j + 1, X.get(i + 1, j + 1) - means[j])
    return result


def covariance_matrix(X_centered: MMatrix) -> MMatrix:
    n = X_centered.n
    m = X_centered.m
    if n <= 1:
        raise ValueError("Нужно больше одной строки для вычисления ковариационной матрицы.")
    X_T = X_centered.transpose()
    C = X_T.multiply(X_centered)
    for i in range(C.n):
        for j in range(C.m):
            C.set(i + 1, j + 1, C.get(i + 1, j + 1) / (n - 1))
    return C


def gauss_solver(A: MMatrix, b: MMatrix) -> List[MMatrix]:
    n = A.n
    m = A.m
    if n != m:
        raise ValueError("Матрица A должна быть квадратной (n × n) для решения системы.")

    # Создаем расширенную матрицу [A|b]
    augmented_matrix = [A[i][:] + [b[i][0]] for i in range(n)]

    # Прямой ход метода Гаусса с выбором ведущего элемента
    for i in range(n):
        # Поиск максимального элемента в столбце i
        max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        if abs(augmented_matrix[max_row][i]) < 1e-12:
            continue  # Пропустить, если столбец нулевой

        # Обмен строк
        augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        # Исключение элементов ниже ведущего
        pivot = augmented_matrix[i][i]
        if pivot == 0:
            continue
        for j in range(i + 1, n):
            factor = augmented_matrix[j][i] / pivot
            for k in range(i, m + 1):
                augmented_matrix[j][k] -= factor * augmented_matrix[i][k]

    # Обратный ход Гаусса
    solutions = []
    tol = 1e-6

    # Определение ранга матрицы
    rank = 0
    for i in range(n):
        if any(abs(augmented_matrix[i][j]) > tol for j in range(m)):
            rank += 1
    # Определение ведущих переменных
    lead_vars = [-1] * m
    for i in range(rank):
        for j in range(m):
            if abs(augmented_matrix[i][j]) > tol:
                lead_vars[j] = i
                break

    # Сбор свободных переменных
    free_vars = [j for j in range(m) if lead_vars[j] == -1]

    if not free_vars:
        # Единственное решение
        solution = MMatrix(m, 1)
        for i in range(rank):
            for j in range(m):
                if abs(augmented_matrix[i][j]) > tol:
                    # Вычисляем значение переменной j
                    sum_val = augmented_matrix[i][m]
                    for k in range(j + 1, m):
                        sum_val -= augmented_matrix[i][k] * solution.get(k + 1, 1)
                    val = sum_val / augmented_matrix[i][j]
                    solution.set(j + 1, 1, val)
                    break
            solutions.append(solution)
    else:
        # Построение базиса решений
        for free in free_vars:
            vec = MMatrix(m, 1)
            vec.set(free + 1, 1, 1.0)  # Свободная переменная = 1
            # Обратный ход для определения ведущих переменных
            for i in range(rank - 1, -1, -1):
                lead_col = -1
                for j in range(m):
                    if abs(augmented_matrix[i][j]) > tol:
                        lead_col = j
                        break
                if lead_col == -1:
                    continue
                sum_ax = 0.0
                for k in range(lead_col + 1, m):
                    sum_ax += augmented_matrix[i][k] * vec.get(k + 1, 1)
                val = (augmented_matrix[i][m] - sum_ax) / augmented_matrix[i][lead_col]
                vec.set(lead_col + 1, 1, val)

            solutions.append(vec)

    return solutions


def gauss_determinant(M):
    n = M.n
    A = [[M.get(i + 1, j + 1) for j in range(n)] for i in range(n)]
    det_sign = 1
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[max_row][i]) < 1e-12:
            return 0
        if i != max_row:
            A[i], A[max_row] = A[max_row], A[i]
            det_sign *= -1
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
    det = det_sign
    for i in range(n):
        det *= A[i][i]
    return det


def gershgorin_bounds(matrix: MMatrix) -> tuple:
    n = matrix.n
    lower = float('inf')
    upper = -float('inf')
    for i in range(n):
        radius = 0.0
        for j in range(n):
            if j != i:
                radius += abs(matrix.get(i + 1, j + 1))
        center = matrix.get(i + 1, i + 1)
        current_lower = center - radius
        current_upper = center + radius
        lower = min(lower, current_lower)
        upper = max(upper, current_upper)
    return lower, upper


def determinant_at_lambda(matrix: MMatrix, lambda_val: float) -> float:
    n = matrix.n
    C_minus_lambda = MMatrix(n, n)
    for i in range(n):
        for j in range(n):
            value = matrix.get(i + 1, j + 1) - (lambda_val if i == j else 0)
            C_minus_lambda.set(i + 1, j + 1, value)
    return gauss_determinant(C_minus_lambda)


def find_eigenvalues(C: MMatrix, tol: float = 1e-6) -> List[float]:
    lower, upper = gershgorin_bounds(C)
    search_step = max((upper - lower) / 300, tol)
    eigenvalues = []
    current_lambda = lower
    prev_det = determinant_at_lambda(C, current_lambda)
    while current_lambda <= upper:
        current_lambda += search_step
        current_det = determinant_at_lambda(C, current_lambda)
        if prev_det * current_det < 0 or abs(current_det) < tol:
            a = current_lambda - search_step
            b = current_lambda
            while (b - a) > tol:
                mid = (a + b) / 2
                mid_det = determinant_at_lambda(C, mid)
                if mid_det * determinant_at_lambda(C, a) < 0:
                    b = mid
                else:
                    a = mid
            eigenvalues.append((a + b) / 2)
            prev_det = current_det

        decimal_places = int(-math.log10(tol))
        unique_eigs = {round(eig, decimal_places) for eig in eigenvalues}
    return sorted(unique_eigs, reverse=True)


def normalize(vector: MMatrix) -> int:
    norm = 0.0
    for i in range(vector.n):
        norm += vector.get(i + 1, 1) ** 2
    return math.sqrt(norm)


def find_eigenvectors(C: 'MMatrix', eigenvalues: List[float]) -> List['MMatrix']:
    n = C.n
    eigenvectors = []
    unique_vectors = []  # Список кортежей для проверки уникальности

    for eigenvalue in eigenvalues:
        modified_matrix = MMatrix(n, n)
        for i in range(n):
            for j in range(n):
                value = C.get(i + 1, j + 1)
                if i == j:
                    value -= eigenvalue
                modified_matrix.set(i + 1, j + 1, value)

        result_vector = MMatrix(n, 1)
        solutions = gauss_solver(modified_matrix, result_vector)
        for vec in solutions:
            # Нормализация вектора
            norm = 0.0
            for i in range(n):
                norm += vec.get(i + 1, 1) ** 2
            norm = math.sqrt(norm)
            if norm < 1e-10:
                continue
            # Создаем кортеж с округленными значениями (6 знаков)
            vec_tuple = tuple(round(vec.get(i + 1, 1), 6) for i in range(n))
            # Проверяем уникальность
            if vec_tuple not in unique_vectors:
                unique_vectors.append(vec_tuple)
                normalized_vec = MMatrix(n, 1)
                for i in range(n):
                    normalized_vec.set(i + 1, 1, vec.get(i + 1, 1) / norm)
                eigenvectors.append(normalized_vec)

    return eigenvectors


def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    if k > len(eigenvalues) or k <= 0:
        raise ValueError("k должно быть в пределах от 1 до числа собственных значений")
    total_sum = 0.0
    top_sum = 0.0
    for i in range(len(eigenvalues)):
        total_sum += eigenvalues[i]
        if i < k:
            top_sum += eigenvalues[i]
    if total_sum == 0:
        return 0.0
    return top_sum / total_sum
