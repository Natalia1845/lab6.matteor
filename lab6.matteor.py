import numpy as np

# Матриця втрат F
F = np.array([
    [5, 10, 18, 25],
    [8, 7, 20, 22],
    [21, 18, 12, 5],
    [15, 12, 25, 15]
])

# Вагові коефіцієнти для кожного стану
weights = [0.25, 0.25, 0.25, 0.25]  # Приклад рівномірних вагових коефіцієнтів

# Функція для нормалізації
def normalize_matrix(F, method="min_square"):
    if method == "min_square":
        return np.array([[-(x**2) for x in row] for row in F])
    elif method == "relative":
        return np.array([[x / F.min() for x in row] for row in F])

# Розрахунок критерію Бернуллі-Лапласа (середні втрати)
def bernoulli_laplace_criterion(F, weights):
    return np.dot(F, weights)

# Розрахунок критерію Вальда (мінімізація максимальних втрат)
def wald_criterion(F):
    return np.min(np.max(F, axis=1))

# Розрахунок для варіантів
# Варіант 1: Метод мінімальних квадратів, критерій Бернуллі-Лапласа
F_norm_1 = normalize_matrix(F, method="min_square")
laplace_result_1 = bernoulli_laplace_criterion(F_norm_1, weights)

# Варіант 2: Метод мінімальних квадратів, критерій Вальда
wald_result_1 = wald_criterion(F_norm_1)

# Варіант 3: Відносна нормалізація, критерій Бернуллі-Лапласа
F_norm_3 = normalize_matrix(F, method="relative")
laplace_result_3 = bernoulli_laplace_criterion(F_norm_3, weights)

# Варіант 4: Відносна нормалізація, критерій Вальда
wald_result_3 = wald_criterion(F_norm_3)

# Виведення результатів
print("Варіант 1: Мінімальні квадрати, Бернуллі-Лаплас:", laplace_result_1)
print("Варіант 2: Мінімальні квадрати, Вальд:", wald_result_1)
print("Варіант 3: Відносна нормалізація, Бернуллі-Лаплас:", laplace_result_3)
print("Варіант 4: Відносна нормалізація, Вальд:", wald_result_3)
