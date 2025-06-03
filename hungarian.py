import numpy as np


def hungarian_algorithm(cost_matrix):
    """
    Реализация Венгерского алгоритма для задачи о назначениях

    Параметры:
    cost_matrix (list[list]): Квадратная матрица стоимостей n x n

    Возвращает:
    tuple: (assignments, total_cost)
      - assignments: Список кортежей (работник, задача)
      - total_cost: Общая минимальная стоимость
    """
    # Конвертируем в numpy массив для удобства
    matrix = np.array(cost_matrix, dtype=float)
    n = matrix.shape[0]

    # Шаг 1: Редукция строк
    for i in range(n):
        min_val = np.min(matrix[i])
        matrix[i] -= min_val

    # Шаг 2: Редукция столбцов
    for j in range(n):
        min_val = np.min(matrix[:, j])
        matrix[:, j] -= min_val

    # Инициализация переменных
    assignments = []
    total_cost = 0

    # Главный цикл алгоритма
    while len(assignments) < n:
        # Шаг 3: Поиск максимального паросочетания по нулевым элементам
        zeros = []
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 0:
                    zeros.append((i, j))

        # Поиск назначений (жадный алгоритм)
        row_covered = set()
        col_covered = set()
        current_assignments = []

        # Сортируем нули по количеству нулей в строке и столбце
        zeros.sort(key=lambda pos: (np.sum(matrix[pos[0]] == 0),
                                    np.sum(matrix[:, pos[1]] == 0)))

        for i, j in zeros:
            if i not in row_covered and j not in col_covered:
                current_assignments.append((i, j))
                row_covered.add(i)
                col_covered.add(j)

        # Если найдено полное паросочетание
        if len(current_assignments) == n:
            assignments = current_assignments
            break

        # Шаг 4: Покрытие нулей минимальным числом линий
        covered_rows = set(range(n)) - row_covered
        covered_cols = set()

        # Поиск столбцов с нулями в непокрытых строках
        for i in covered_rows:
            for j in range(n):
                if matrix[i, j] == 0:
                    covered_cols.add(j)

        # Поиск строк с назначениями в покрытых столбцах
        new_covered_rows = set()
        for j in covered_cols:
            for i, j_assign in current_assignments:
                if j == j_assign:
                    new_covered_rows.add(i)

        covered_rows |= new_covered_rows

        # Шаг 5: Корректировка матрицы
        # Находим минимальный непокрытый элемент
        min_val = float('inf')
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    if matrix[i, j] < min_val:
                        min_val = matrix[i, j]

        # Обновляем матрицу
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    matrix[i, j] -= min_val
                elif i in covered_rows and j in covered_cols:
                    matrix[i, j] += min_val

    # Вычисляем общую стоимость
    for i, j in assignments:
        total_cost += cost_matrix[i][j]

    return assignments, total_cost


# Матрица стоимостей 3x3
cost_matrix = [
    [9, 2, 7],
    [6, 4, 3],
    [5, 8, 1]
]

# Решение задачи
assignments, total_cost = hungarian_algorithm(cost_matrix)

# Вывод результатов
print("Оптимальное назначение:")
for worker, task in assignments:
    print(f"Работник {worker} → Задача {task} (стоимость: {cost_matrix[worker][task]})")

print(f"Общая стоимость: {total_cost}")