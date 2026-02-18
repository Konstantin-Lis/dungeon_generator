import random
from PIL import Image








def perlin_noise_1_frequency(map_size: tuple[int, int], frequency: int):
    """
    Сначала создаём карту размером побольше и заполняем её
    Нам надо с гарантией вне зависимости от остатка покрыть карту шумом, поэтому делаем +2
    И далее определяем размеры большой карты по частоте
    """
    x_main_points_num = map_size[0] // frequency + 2
    y_main_points_num = map_size[1] // frequency + 2

    big_map_size = (
        (x_main_points_num - 1) * frequency + 1,
        (y_main_points_num - 1) * frequency + 1,
    )

    big_map_heights = [[0.0 for _ in range(big_map_size[1] + 1)] for _ in range(big_map_size[0] + 1)]

    # Теперь расставляем в каждую из главных точек вектора
    vectors = [[[random.random() - 0.5, random.random() - 0.5] for _ in range(y_main_points_num)]
               for _ in range(x_main_points_num)]

    # Заполняем матрицу значений
    for x_point in range(x_main_points_num - 1):
        for y_point in range(y_main_points_num - 1):
            for place_x in range(frequency):
                for place_y in range(frequency):
                    z_result = 0
                    # Каждую точку нашего прямоугольника высчитываем исходя из 4-х угловых точек прямоугольника
                    # Первое слагаемое - вклад коэффициента по х, второе - по y
                    zx_0_0 = vectors[x_point][y_point][0] * place_x / frequency
                    zy_0_0 = vectors[x_point][y_point][1] * place_y / frequency
                    zx_0_1 = vectors[x_point][y_point + 1][0] * place_x / frequency
                    zy_0_1 = (-1) * vectors[x_point][y_point + 1][1] * (frequency - place_y) / frequency
                    zx_1_0 = (-1) * vectors[x_point + 1][y_point][0] * (frequency - place_x) / frequency
                    zy_1_0 = vectors[x_point + 1][y_point][1] * place_y / frequency
                    zx_1_1 = (-1) * vectors[x_point + 1][y_point + 1][0] * (frequency - place_x) / frequency
                    zy_1_1 = (-1) * vectors[x_point + 1][y_point + 1][1] * (frequency - place_y) / frequency

                    z_0_0 = zx_0_0 + zy_0_0
                    z_0_1 = zx_0_1 + zy_0_1
                    z_1_0 = zx_1_0 + zy_1_0
                    z_1_1 = zx_1_1 + zy_1_1

                    z_result += z_0_0 * (frequency - place_x) * (frequency - place_y) / frequency ** 2
                    z_result += z_0_1 * (frequency - place_x) * place_y / frequency ** 2
                    z_result += z_1_0 * place_x * (frequency - place_y) / frequency ** 2
                    z_result += z_1_1 * place_x * place_y / frequency ** 2

                    big_map_heights[x_point * frequency + place_x][y_point * frequency + place_y] += z_result

    # Обрезаем матрицу значений до размера карты
    map_heights = []
    for x in range(map_size[0]):
        map_heights.append(big_map_heights[x][:map_size[1]])

    return map_heights


def perlin_noise(seed: int, map_size: tuple[int, int], frequencies: tuple):
    random.seed(seed)

    # Создаём итоговую матрицу и заполняем её нулями
    map_heights = [[0.0 for _ in range(map_size[1])] for _ in range(map_size[0])]

    # Потихоньку добавляем к ней созданные слои
    for freq in frequencies:
        map_part = perlin_noise_1_frequency(map_size, freq)
        for x in range(map_size[0]):
            for y in range(map_size[1]):
                map_heights[x][y] += map_part[x][y]

    return map_heights


def normalize_map(map_heights):
    arr_shape = (len(map_heights), len(map_heights[0]))

    # Находим минимальный и максимальный элементы, сразу сдвигаем максимальный (мини-оптимизация)
    min_el = map_heights[0][0]
    max_el = map_heights[0][0]
    for row in map_heights:
        for el in row:
            if min_el > el:
                min_el = el
            if max_el < el:
                max_el = el
    max_el -= min_el

    # Сдвигаем все элементы, чтобы минимум был в 0 и делим на максимум, ставя его в 1
    # Потом возводим в квадрат (для чего-то) или не возводим, я хз, нихрена не работает
    for x_chor in range(arr_shape[0]):
        for y_chor in range(arr_shape[1]):
            map_heights[x_chor][y_chor] -= min_el
            map_heights[x_chor][y_chor] /= max_el
            map_heights[x_chor][y_chor] **= 2

    return map_heights


def heights_to_maze(map_heights, difference: float):
    """
    Функция преобразует картту высот в карту лабиринта

    :param map_heights: Созданная ранее карта
    :param difference: Параметр для определения границы (стена - клетка)
    :return:
    """
    arr_shape = (len(map_heights), len(map_heights[0]))
    map_maze = [[0 for _ in range(arr_shape[1])] for _ in range(arr_shape[0])]
    for x in range(arr_shape[0]):
        for y in range(arr_shape[1]):
            if map_heights[x][y] > difference:
                map_maze[x][y] = 1
            else:
                map_maze[x][y] = 0

    return map_maze


def draw_heights(map_heights):
    arr_shape = (len(map_heights), len(map_heights[0]))
    img = Image.new('RGB', arr_shape)
    for x in range(arr_shape[0]):
        for y in range(arr_shape[1]):
            gray_value = int(map_heights[x][y] * 255)
            color = (gray_value, gray_value, gray_value)
            img.putpixel((x, y), color)
    return img


if __name__ == '__main__':
    seed = 2

    base_map = perlin_noise(1, (100, 30), (5, 8, 13, 20))
    normal_map = normalize_map(base_map)
    map_maze = heights_to_maze(normal_map, 0.2)

    image_maze = draw_heights(map_maze)
    image_maze.save(f'pictures/image_maze_{seed}.png')
    image_maze.close()

    image_map = draw_heights(normal_map)
    image_map.save(f'pictures/image_map_{seed}.png')
    image_map.close()
