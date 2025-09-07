# test_renderer.py
"""
Модульный тест для renderer.py v9.2
-----------------------------------------
- Создает искусственный набор данных, имитирующий результат симуляции
  с двумя стабильными квазичастицами ('комками').
- Вызывает render_frame_worker для создания одного тестового кадра.
- Позволяет быстро проверить и настроить визуализацию, не запуская
  полноценную симуляцию.

Запуск: python test_renderer.py
Результат: файл 'test_render.png' в текущей папке.
"""
import numpy as np
import os
from termcolor import cprint

# --- Импортируем то, что тестируем ---
from topologies import CrystalTopology
from renderer import render_frame_worker

def create_test_data(topology):
    """Генерирует 'идеальные' данные для теста."""
    num_points = topology.num_points
    points = topology.points

    # --- 1. Создаем фон: "холодный голубой вакуум" ---
    psi = np.ones(num_points, dtype=complex) * 0.1 * np.exp(1j * np.pi)

    # --- 2. Определяем центры и размеры наших двух "комков" ---
    center1 = np.array([topology.width * 0.25, topology.height * np.sqrt(3)/2 * 0.6])
    center2 = np.array([topology.width * 0.75, topology.height * np.sqrt(3)/2 * 0.3])
    radius1 = 6.0
    radius2 = 8.0

    # --- 3. Находим узлы, принадлежащие каждому "комку" ---
    dist1 = np.linalg.norm(points - center1, axis=1)
    dist2 = np.linalg.norm(points - center2, axis=1)

    nodes1 = np.where(dist1 < radius1)[0]
    nodes2 = np.where(dist2 < radius2)[0]

    # --- 4. Задаем свойства "комков" ---
    # Комок 1: высокая амплитуда, "антиферромагнитная" фаза
    amplitudes = np.abs(psi)
    amplitudes[nodes1] = np.exp(-(dist1[nodes1]**2) / (2 * (radius1/2)**2)) * 0.9 + 0.1

    phases = np.angle(psi)
    # Создаем чередующуюся ("шахматную") фазу
    for node_idx in nodes1:
        x, y = topology.points[node_idx]
        # Простое условие чередования
        if (int(x) + int(y / (np.sqrt(3)/2))) % 2 == 0:
            phases[node_idx] = 0.0 # Красный
        else:
            phases[node_idx] = np.pi # Голубой

    # Комок 2: высокая амплитуда, "ортогональная" когерентная фаза (желто-зеленая)
    amplitudes[nodes2] = np.exp(-(dist2[nodes2]**2) / (2 * (radius2/2)**2)) * 0.8 + 0.1
    phases[nodes2] = np.pi / 2 # Зеленый

    # Собираем финальное поле psi
    psi = amplitudes * np.exp(1j * phases)

    # --- 5. Создаем "отчет трекера" (stable_particles) ---
    # Это то, что наш tracker.py v3.0 должен был бы найти

    # Характеризуем комок 1
    amp_sq1 = np.abs(psi[nodes1])**2
    mass1 = np.sum(amp_sq1)
    pos1 = np.average(points[nodes1], weights=amp_sq1, axis=0)
    charge1 = np.sum(np.angle(psi[nodes1])) / (2 * np.pi)

    # Характеризуем комок 2
    amp_sq2 = np.abs(psi[nodes2])**2
    mass2 = np.sum(amp_sq2)
    pos2 = np.average(points[nodes2], weights=amp_sq2, axis=0)
    charge2 = np.sum(np.angle(psi[nodes2])) / (2 * np.pi)

    stable_particles = [
        {
            "track_id": 101, "age": 150, "size": len(nodes1),
            "mass": mass1, "average_charge": charge1,
            "position": pos1, "node_indices": nodes1
        },
        {
            "track_id": 102, "age": 120, "size": len(nodes2),
            "mass": mass2, "average_charge": charge2,
            "position": pos2, "node_indices": nodes2
        }
    ]

    return {
        "points": points,
        "psi": psi,
        "simplices": topology.get_simplices(),
        "stable_particles": np.array(stable_particles, dtype=object),
        "tracked_count": 2
    }


if __name__ == "__main__":
    cprint("--- Running Renderer Unit Test ---", "cyan")

    # 1. Создаем топологию
    test_topology = CrystalTopology(width=30, height=20)

    # 2. Генерируем тестовые данные
    cprint("Generating artificial data with two quasi-particles...", "yellow")
    test_data = create_test_data(test_topology)

    # 3. Сохраняем их во временный .npz файл (имитируем работу main.py)
    temp_data_path = "temp_test_data.npz"
    np.savez_compressed(temp_data_path, **test_data)

    # 4. Готовим аргументы для render_frame_worker
    args_tuple = (
        9999, # Номер кадра
        temp_data_path,
        ".", # Сохранить в текущую папку
        {'seed': 'TEST', 'topology': '30x20', 'ic': 'test_case'}
    )

    # 5. Вызываем рендерер!
    cprint("Calling render_frame_worker...", "yellow")
    output_filename = os.path.join(".", "frame_9999.png")

    # Переименовываем для понятности
    final_filename = "test_render.png"
    if os.path.exists(final_filename):
        os.remove(final_filename)

    result = render_frame_worker(args_tuple)

    # 6. Проверяем результат и убираем за собой
    if result is None and os.path.exists(output_filename):
        os.rename(output_filename, final_filename)
        cprint(f"SUCCESS! Test frame saved as '{final_filename}'. Please review it.", "green")
    else:
        cprint(f"FAILURE! Renderer returned an error: {result}", "red")

    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
    if os.path.exists(output_filename): # На случай если переименование не удалось
        os.remove(output_filename)

    cprint("--- Test Complete ---", "cyan")
