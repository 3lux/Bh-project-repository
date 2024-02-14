import numpy as np
import pandas as pd

# 1) Создание матрицы случайных чисел, сохранение в файл и чтение из файла
matrix = np.random.randint(0, 10, size=(3, 3))
np.savetxt('matrix.txt', matrix, fmt='%d')
read_matrix = np.loadtxt('matrix.txt', dtype=int)
print("1) Матрица:")
print(read_matrix)

# 2) Чтение диагональных элементов матрицы
diagonal_elements = np.diag(read_matrix)
print("\n2) Диагональные элементы:")
print(diagonal_elements)

# 3) Объединение двух массивов в один упорядоченный
array1 = np.array([3, 1, 4])
array2 = np.array([2, 7, 5])
merged_array = np.sort(np.concatenate((array1, array2)))
print("\n3) Объединенный и упорядоченный массив:")
print(merged_array)

# 4) Вставка нулевых элементов после отрицательных элементов
array_with_zeros = np.insert(read_matrix, np.where(read_matrix < 0)[0] + 1, 0, axis=1)
print("\n4) Массив с нулями после отрицательных элементов:")
print(array_with_zeros)

# 5) Формирование нового массива по правилу
array_a = np.array([1, 2, 3, 4, 5])
array_b = np.array([(array_a[k:].mean()) for k in range(len(array_a))])
print("\n5) Новый массив b:")
print(array_b)

# 6) Замена каждого элемента на среднее арифметическое соседей
array_c = np.array([1, 2, 3, 4, 5])
array_c[1:-1] = (array_c[:-2] + array_c[1:-1] + array_c[2:]) / 3
print("\n6) Массив с заменой элементов:")
print(array_c)

# 7) Создание матрицы 3x4 с одной единицей в строке
matrix_7 = np.zeros((3, 4))
matrix_7[np.arange(3), np.random.randint(4, size=3)] = 1
print("\n7) Матрица:")
print(matrix_7)

# 8) Вывод нечетных строк с первым элементом больше последнего
matrix_8 = np.random.randint(0, 10, size=(40, 4))
odd_rows = matrix_8[(np.arange(matrix_8.shape[0]) % 2 != 0) & (matrix_8[:, 0] > matrix_8[:, -1])]
print("\n8) Нечетные строки:")
print(odd_rows)

# 9) Подсчет количества встречающихся чисел 7
matrix_9 = np.random.randint(0, 10, size=(5, 5))
count_sevens = np.sum(matrix_9 == 7)
print("\n9) Количество встречающихся чисел 7:")
print(count_sevens)

# 10) Формирование квадратной матрицы
n = 5
diagonal_values = np.random.randint(1, 10, size=n)
matrix_10 = np.eye(n) * diagonal_values
print("\n10) Квадратная матрица:")
print(matrix_10)

# 11) Удаление строк без повторяющихся элементов
matrix_11 = np.random.randint(1, 5, size=(5, 5))
unique_rows = matrix_11[(matrix_11[:, :] == matrix_11[:, None, :]).sum(axis=-1) > 1]
print("\n11) Матрица после удаления строк:")
print(unique_rows)

# 12) Замена строк и столбцов матрицы
matrix_12 = np.random.randint(0, 10, size=(3, 3))
matrix_12[[0, -1]] = matrix_12[[-1, 0]]
matrix_12[:, [0, -1]] = matrix_12[:, [-1, 0]]
print("\n12) Матрица после замены строк и столбцов:")
print(matrix_12)

# 13) Произведение двух матриц и внутреннее/внешнее произведение строк
matrix_a = np.random.randint(1, 10, size=(2, 2))
matrix_b = np.random.randint(1, 10, size=(2, 2))
product_ab = np.dot(matrix_a, matrix_b)
inner_product = np.inner(matrix_a[0], matrix_b[0]), np.inner(matrix_a[1], matrix_b[1])
outer_product = np.outer(matrix_a[0], matrix_b[0]), np.outer(matrix_a[1], matrix_b[1])

print("\n13) Произведение двух матриц:")
print(product_ab)
print("\nВнутреннее произведение строк:")
print(inner_product)
print("\nВнешнее произведение строк:")
print(outer_product)

# 14) Сортировка строк матрицы по произведению элементов
matrix_14 = np.random.randint(1, 10, size=(3, 3))
sorted_matrix_14 = matrix_14[np.argsort(np.prod(matrix_14, axis=1))]
print("\n14) Отсортированные строки по произведению элементов:")
print(sorted_matrix_14)

# 15) Обнуление элементов выше главной и ниже побочной диагонали
matrix_15 = np.random.randint(1, 10, size=(5, 5))
upper_triangle_indices = np.triu_indices(matrix_15.shape[0], k=1)
lower_triangle_indices = np.tril_indices(matrix_15.shape[0], k=-1)
matrix_15[upper_triangle_indices] = 0
matrix_15[lower_triangle_indices] = 0
print("\n15) Матрица с обнуленными элементами:")
print(matrix_15)

# 16) Создание DataFrame с оценками
subjects = pd.Series(['Математика', 'Чтение', 'Физика'])
teachers = pd.Series(['Иванов', 'Петров', 'Сидоров'])
grades = pd.Series([5, 4, 3])
df = pd.DataFrame({'Предмет': subjects, 'Преподаватель': teachers, 'Оценка': grades})
print("\n16) DataFrame:")
print(df)

# 17) Изменение названий столбцов и перемешивание строк
df.columns = ['Предмет', 'Преподаватель', 'Оценка']
df = df.sample(frac=1).reset_index(drop=True)
print("\n17) Измененный DataFrame:")
print(df)

# 18) Назначение столбца предметов в качестве индекса
df.set_index('Предмет', inplace=True)
print("\n18) Общая информация о DataFrame:")
print(df.info())
print("\nРазмер массива:")
print(df.shape)
print("\nСписок названий строк:")
print(df.index.tolist())
print("\nСписок названий столбцов:")
print(df.columns.tolist())
print("\nСредняя оценка по всем предметам:")
print(df['Оценка'].mean())

# 19) Создание массива и его преобразование в DataFrame с добавлением столбца Total
array_19 = np.random.randint(0, 10, size=(10, 10))
df_19 = pd.DataFrame(array_19)
df_19.columns = [f'Col_{i}' for i in range(1, 11)]
df_19['Total'] = df_19.sum(axis=1)
print("\n19) DataFrame с добавленным столбцом Total:")
print(df_19)

# 20) Применение функции к каждому элементу и замена NaN на 0
df_20 = df_19.applymap(lambda x: (x * x + 2) ** 0.5)
df_20.fillna(0, inplace=True)
print("\n20) DataFrame после преобразования:")
print(df_20)
