import matplotlib.pyplot as plt

from matrix import MMatrix
from matplotlib.figure import Figure
from light import *
from normal import *


def pca(X: 'Matrix', k: int) -> tuple['Matrix', float]:
    """
    Вход:
    X: матрица данных (n×m)
    k: число главных компонент
    Выход:
    X_proj: проекция данных (n×k)
    : доля объяснённой дисперсии
    """
    X = center_data(X)
    X = covariance_matrix(X)
    eigen_values = find_eigenvalues(X)
    eigen_vectors = find_eigenvectors(X, eigen_values)
    main_comp = eigen_vectors[:k]
    Vk = MMatrix(X.n, k)
    for i in range(X.n):
        for j in range(k):
            Vk.element_add(i + 1, j + 1, main_comp[j].get(i + 1, 1))
    return X.multiply(Vk), explained_variance_ratio(eigen_values, k)


def plot_pca_projection(X_proj: MMatrix) -> Figure:
    # Извлекаем координаты из матрицы проекции (n x 2)
    x = [X_proj.get(i + 1, 1) for i in range(X_proj.n)]
    y = [X_proj.get(i + 1, 2) for i in range(X_proj.n)]

    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(10, 7))

    # Рисуем точки
    ax.scatter(x, y, s=50, linewidth=0.8)

    # Добавляем подписи точек
    for i in range(len(x)):
        ax.text(x[i] + 0.1, y[i] + 0.1, str(i + 1), color='darkred')

    # Настраиваем оформление
    ax.set_title("Проекция данных на первые 2 главные компоненты")
    ax.set_xlabel("PC1 (Главная компонента 1)")
    ax.set_ylabel("PC2 (Главная компонента 2)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)

    return fig


def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    s = 0
    for i in range(X_recon.n):
        for j in range(X_recon.m):
            s += (X_orig.get(i + 1, j + 1) - X_recon.get(i + 1, j + 1)) ** 2
    return s / (X_orig.n * X_orig.m)


# Тестовая матрица
test_matrix = MMatrix(4, 4)
for i in range(4):
    for j in range(4):
        test_matrix.set(i + 1, j + 1, i * j)
print(pca(test_matrix, 2))

X_proj, _ = pca(test_matrix, 2)
# Визуализация
fig = plot_pca_projection(X_proj)
plt.show()
print(reconstruction_error(test_matrix, X_proj))