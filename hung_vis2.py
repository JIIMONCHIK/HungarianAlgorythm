import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import matplotlib.gridspec as gridspec


class HungarianVisualizer:
    def __init__(self, cost_matrix):
        self.cost_matrix = np.array(cost_matrix)
        self.n = len(cost_matrix)
        self.original_matrix = self.cost_matrix.copy()
        self.steps = []
        self.assignment = []
        self.fig = None
        self.ax_matrix = None
        self.ax_graph = None

    def reduce_rows(self):
        """Шаг 1: Редукция строк"""
        row_mins = np.min(self.cost_matrix, axis=1)
        self.cost_matrix = self.cost_matrix - row_mins[:, np.newaxis]
        self._add_step("Редукция строк")

    def reduce_columns(self):
        """Шаг 2: Редукция столбцов"""
        col_mins = np.min(self.cost_matrix, axis=0)
        self.cost_matrix = self.cost_matrix - col_mins
        self._add_step("Редукция столбцов")

    def _add_step(self, description):
        """Сохраняем текущее состояние для визуализации"""
        zeros = []
        for i in range(self.n):
            for j in range(self.n):
                if self.cost_matrix[i, j] == 0:
                    zeros.append((i, j))

        # Текущее паросочетание (упрощенный поиск)
        assignment = []
        row_covered = [False] * self.n
        col_covered = [False] * self.n

        # Сначала добавляем нули, которые являются единственными в строке/столбце
        for i in range(self.n):
            row_zeros = [j for j in range(self.n) if self.cost_matrix[i, j] == 0]
            if len(row_zeros) == 1:
                j = row_zeros[0]
                if not col_covered[j]:
                    assignment.append((i, j))
                    row_covered[i] = True
                    col_covered[j] = True

        # Затем добавляем оставшиеся нули
        for i in range(self.n):
            for j in range(self.n):
                if self.cost_matrix[i, j] == 0:
                    if not row_covered[i] and not col_covered[j]:
                        assignment.append((i, j))
                        row_covered[i] = True
                        col_covered[j] = True

        self.steps.append({
            "matrix": self.cost_matrix.copy(),
            "description": description,
            "zeros": zeros,
            "assignment": assignment,
            "covered_rows": set(),
            "covered_cols": set()
        })

    def solve(self):
        """Реализация Венгерского алгоритма"""
        # Шаг 1: Редукция строк
        self.reduce_rows()

        # Шаг 2: Редукция столбцов
        self.reduce_columns()

        # Итерации до нахождения полного назначения
        iteration = 1
        while len(self.steps[-1]["assignment"]) < self.n:
            # Шаг 3: Покрытие нулей минимальными линиями
            current_step = self.steps[-1]
            assignment = current_step["assignment"]

            # Помечаем строки без назначений
            assigned_rows = set(i for i, j in assignment)
            covered_rows = set(range(self.n)) - assigned_rows

            # Ищем столбцы с нулями в помеченных строках
            covered_cols = set()
            for i in covered_rows:
                for j in range(self.n):
                    if self.cost_matrix[i, j] == 0:
                        covered_cols.add(j)

            # Помечаем строки с назначениями в новых столбцах
            new_covered_rows = set()
            for j in covered_cols:
                for i, j_assign in assignment:
                    if j == j_assign:
                        new_covered_rows.add(i)

            covered_rows |= new_covered_rows

            # Шаг 4: Корректировка матрицы
            # Находим минимальный непокрытый элемент
            min_val = float('inf')
            for i in range(self.n):
                for j in range(self.n):
                    if i not in covered_rows and j not in covered_cols:
                        if self.cost_matrix[i, j] < min_val:
                            min_val = self.cost_matrix[i, j]

            # Обновляем матрицу
            for i in range(self.n):
                for j in range(self.n):
                    if i not in covered_rows and j not in covered_cols:
                        self.cost_matrix[i, j] -= min_val
                    elif i in covered_rows and j in covered_cols:
                        self.cost_matrix[i, j] += min_val

            # Сохраняем шаг с информацией о покрытии
            self._add_step(f"Корректировка матрицы (Итерация {iteration})")
            self.steps[-1]["covered_rows"] = covered_rows
            self.steps[-1]["covered_cols"] = covered_cols
            iteration += 1

        # Сохраняем финальное назначение
        self.assignment = self.steps[-1]["assignment"]
        self._add_step("Финальное назначение")
        return self.assignment

    def visualize(self, save_gif=False, filename="hungarian_algorithm.gif"):
        """Создание анимации с визуализацией"""
        self.fig = plt.figure(figsize=(15, 8), dpi=100)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

        self.ax_matrix = plt.subplot(gs[0])
        self.ax_graph = plt.subplot(gs[1])

        self.fig.tight_layout(pad=5.0)
        ani = FuncAnimation(self.fig, self._update, frames=len(self.steps),
                            interval=2000, repeat=False)

        if save_gif:
            try:
                ani.save(filename, writer='pillow', fps=1)
                print(f"Анимация сохранена как '{filename}'")
            except ImportError:
                print("Для сохранения GIF установите pillow: pip install pillow")

        plt.show()
        return ani

    def _update(self, frame):
        """Обновление кадра анимации"""
        for ax in [self.ax_matrix, self.ax_graph]:
            ax.clear()
            ax.set_facecolor('#f0f0f0')

        step = self.steps[frame]
        matrix = step["matrix"]
        zeros = step["zeros"]
        assignment = step["assignment"]
        description = step["description"]
        covered_rows = step["covered_rows"]
        covered_cols = step["covered_cols"]

        # Визуализация матрицы
        self._draw_matrix(matrix, zeros, assignment, covered_rows, covered_cols)

        # Визуализация графа
        self._draw_graph(assignment)

        # Общий заголовок
        self.fig.suptitle(f"Шаг {frame + 1}/{len(self.steps)}: {description}",
                          fontsize=16, fontweight='bold')

    def _draw_matrix(self, matrix, zeros, assignment, covered_rows, covered_cols):
        """Отрисовка матрицы стоимостей"""
        # Создаем цветовую карту
        cmap = plt.cm.Blues
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # Рисуем ячейки матрицы
        cell_size = 1.0
        rectangles = []
        colors_list = []

        for i in range(self.n):
            for j in range(self.n):
                # Определяем цвет ячейки
                value = matrix[i, j]
                color = cmap(norm(value))

                # Если это назначение - делаем цвет ярче
                if (i, j) in assignment:
                    color = plt.cm.Reds(norm(value))

                # Если это ноль - добавляем рамку
                edgecolor = 'red' if (i, j) in zeros else 'black'
                linewidth = 2.5 if (i, j) in zeros else 1

                # Если ячейка покрыта - добавляем штриховку
                hatch = None
                if i in covered_rows or j in covered_cols:
                    hatch = '////'

                rect = Rectangle((j, self.n - i - 1), cell_size, cell_size,
                                 edgecolor=edgecolor, linewidth=linewidth,
                                 hatch=hatch)
                rectangles.append(rect)
                colors_list.append(color)

        # Создаем коллекцию прямоугольников
        pc = PatchCollection(rectangles, facecolor=colors_list,
                             edgecolor=[r.get_edgecolor() for r in rectangles],
                             linewidth=[r.get_linewidth() for r in rectangles])
        self.ax_matrix.add_collection(pc)

        # Добавляем значения
        for i in range(self.n):
            for j in range(self.n):
                value = matrix[i, j]
                color = 'black' if value > vmin + (vmax - vmin) * 0.4 else 'white'
                fontweight = 'bold' if (i, j) in assignment else 'normal'
                self.ax_matrix.text(j + 0.5, self.n - i - 0.5, f"{value:.1f}",
                                    ha='center', va='center',
                                    color=color, fontsize=12, fontweight=fontweight)

        # Настройка осей
        self.ax_matrix.set_xlim(0, self.n)
        self.ax_matrix.set_ylim(0, self.n)
        self.ax_matrix.set_xticks(np.arange(0.5, self.n, 1))
        self.ax_matrix.set_yticks(np.arange(0.5, self.n, 1))
        self.ax_matrix.set_xticklabels([f"Задача {j}" for j in range(self.n)])
        self.ax_matrix.set_yticklabels([f"Работник {self.n - i - 1}" for i in range(self.n)])
        self.ax_matrix.set_title("Матрица стоимостей", fontsize=14)
        self.ax_matrix.grid(True, linestyle='--', alpha=0.7)
        self.ax_matrix.tick_params(axis='both', which='both', length=0)

        # Добавляем легенду покрытия
        if covered_rows or covered_cols:
            self.ax_matrix.text(0.5, -0.5, "Заштрихованные области: покрытые строки/столбцы",
                                fontsize=10, ha='center', transform=self.ax_matrix.transAxes)

    def _draw_graph(self, assignment):
        """Отрисовка графа назначений"""
        # Параметры визуализации
        worker_x = 1
        task_x = 5
        node_radius = 0.3
        worker_colors = ['#FFCC66', '#FFDD99', '#FFEEBB', '#FFFFCC', '#FFF0CC']
        task_colors = ['#66CCFF', '#99DDFF', '#BBEEFF', '#CCFFFF', '#CCF0FF']

        # Рисуем работников
        for i in range(self.n):
            y = self.n - i - 1
            color = worker_colors[i % len(worker_colors)]
            circle = Circle((worker_x, y), node_radius, color=color, ec='black', lw=2)
            self.ax_graph.add_patch(circle)
            self.ax_graph.text(worker_x, y, f"W{i}",
                               ha='center', va='center', fontsize=12, fontweight='bold')

        # Рисуем задачи
        for j in range(self.n):
            y = self.n - j - 1
            color = task_colors[j % len(task_colors)]
            circle = Circle((task_x, y), node_radius, color=color, ec='black', lw=2)
            self.ax_graph.add_patch(circle)
            self.ax_graph.text(task_x, y, f"T{j}",
                               ha='center', va='center', fontsize=12, fontweight='bold')

        # Рисуем связи
        for i in range(self.n):
            for j in range(self.n):
                y_worker = self.n - i - 1
                y_task = self.n - j - 1
                cost = self.original_matrix[i, j]

                # Стиль линии в зависимости от назначения
                if (i, j) in assignment:
                    color = 'red'
                    linewidth = 3
                    linestyle = '-'
                else:
                    color = 'gray'
                    linewidth = 1
                    linestyle = '--'

                # Рисуем линию
                self.ax_graph.plot([worker_x + node_radius, task_x - node_radius],
                                   [y_worker, y_task],
                                   color=color, linewidth=linewidth, linestyle=linestyle)

                # Подпись стоимости
                if cost > 0:  # Не отображаем стоимость 0 для экономии места
                    mid_x = (worker_x + task_x) / 2
                    mid_y = (y_worker + y_task) / 2
                    self.ax_graph.text(mid_x, mid_y, f"{cost}",
                                       backgroundcolor='white', fontsize=9,
                                       ha='center', va='center', alpha=0.8)

        # Настройка графа
        self.ax_graph.set_xlim(0, 6)
        self.ax_graph.set_ylim(-1, self.n)
        self.ax_graph.set_aspect('equal')
        self.ax_graph.axis('off')
        self.ax_graph.set_title("Граф назначений", fontsize=14)
        self.ax_graph.text(0.5, -0.1, "Красные линии: выбранные назначения\nСерые линии: возможные назначения",
                           transform=self.ax_graph.transAxes, ha='center', fontsize=10)


# Пример использования
if __name__ == "__main__":
    # Матрица стоимостей (работники x задачи)
    cost_matrix = [
        [15, 40, 25, 30],
        [20, 60, 35, 45],
        [10, 30, 20, 25],
        [25, 50, 40, 35]
    ]

    print("Запуск Венгерского алгоритма...")
    print(f"Размер задачи: {len(cost_matrix)}x{len(cost_matrix[0])}")
    print("Исходная матрица стоимостей:")
    for row in cost_matrix:
        print(row)

    # Решение задачи
    visualizer = HungarianVisualizer(cost_matrix)
    assignment = visualizer.solve()

    print("\nОптимальное назначение:")
    total_cost = 0
    for worker, task in assignment:
        cost = cost_matrix[worker][task]
        total_cost += cost
        print(f"Работник {worker} → Задача {task} (стоимость: {cost})")

    print(f"\nОбщая стоимость: {total_cost}")

    # Визуализация
    print("\nЗапуск визуализации...")
    visualizer.visualize(save_gif=True, filename="hungarian_algorithm.gif")