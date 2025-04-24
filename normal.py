from matrix import MMatrix
from light import gauss_solver

import math
from typing import List


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
