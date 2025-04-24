import math
from typing import List


class MMatrix:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.V = [[0] * m for _ in range(n)]

    def __str__(self):  # Простой вывод матрицы в строковом формате
        return "\n".join([" ".join(map(str, row)) for row in self.V])

    def __getitem__(self, idx):  # Получение строки по индексу
        return self.V[idx]

    def __setitem__(self, idx, value):  # Установка строки по индексу
        self.V[idx] = value

    def swap_rows(self, i, j):  # Меняет местами строки i и j
        self.V[i], self.V[j] = self.V[j], self.V[i]

    def element_add(self, i, j, value):  # Добавляет значение в элемент (i, j), индексация с 1
        self.V[i - 1][j - 1] = value

    def get(self, i, j):  # Получение элемента матрицы
        return self.V[i - 1][j - 1]

    def set(self, i, j, value):  # Установка элемента матрицы
        self.V[i - 1][j - 1] = value

    def transpose(self):  # Транспонирование матрицы
        result = MMatrix(self.m, self.n)
        for i in range(self.n):
            for j in range(self.m):
                result.set(j + 1, i + 1, self.get(i + 1, j + 1))
        return result

    def multiply(self, other):  # Умножение матрицы на другую матрицу
        if self.m != other.n:
            raise ValueError("Количество столбцов первой матрицы должно быть равно количеству строк второй.")
        result = MMatrix(self.n, other.m)
        for i in range(self.n):
            for j in range(other.m):
                total = 0
                for k in range(self.m):
                    total += self.get(i + 1, k + 1) * other.get(k + 1, j + 1)
                result.set(i + 1, j + 1, total)
        return result
