# renderer.py v10.2 - "Pro"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
import os
from termcolor import cprint

def render_frame_worker(args_tuple):
    """
    Рабочая функция для одного процесса. Рендерит один кадр симуляции.
    Новое в v10.2:
    - Жестко задано разрешение вывода 1920x1080 (Full HD).
    - Поля вокруг решетки теперь адаптивные для оптимального кадрирования.
    - Улучшена визуальная четкость подсветки частиц.
    """
    frame_num, data_path, frames_dir, shared_info = args_tuple

    try:
        def unwrap(d):
            if hasattr(d, 'ndim') and d.ndim == 0:
                return d.item()
            return d

        data = np.load(data_path, allow_pickle=True)
        points = unwrap(data['points'])
        psi = unwrap(data['psi'])
        simplices = unwrap(data['simplices'])
        stable_particles = unwrap(data['stable_particles'])
        tracked_count = unwrap(data['tracked_count'])

        # --- 1. Инициализация холста для 1080p ---
        # 1920x1080 пикселей. DPI (dots per inch) = разрешение / размер в дюймах.
        # figsize=(16, 9) дюймов. 1920/16 = 120. 1080/9 = 120.
        FIG_WIDTH_INCHES = 16
        FIG_HEIGHT_INCHES = 9
        DPI = 120
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
        ax.set_facecolor('#0d0d0d')
        ax.set_aspect('equal')

        num_points = len(points)
        amplitudes_sq = np.abs(psi)**2

        # --- 2. Отрисовка 'ткани' пространства (треугольников) ---
        patches = []
        face_colors = []
        cmap_bg = plt.get_cmap('magma')

        vmax = np.percentile(amplitudes_sq, 99.8) if num_points > 0 else 1.0 # Чуть выше порог для контраста
        vmin = np.min(amplitudes_sq)
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)

        if simplices is not None and simplices.ndim == 2 and simplices.shape[1] == 3:
            for simplex in simplices:
                if np.any(simplex >= num_points): continue
                patches.append(Polygon(points[simplex], closed=True))
                face_colors.append(cmap_bg(norm(np.mean(amplitudes_sq[simplex]))))

        collection = PatchCollection(patches, facecolors=face_colors, edgecolors='#333333', linewidth=0.5, alpha=0.9, zorder=1)
        ax.add_collection(collection)

        # --- 3. Отрисовка узлов ---
        phases = np.angle(psi)
        node_sizes = amplitudes_sq * 8000 + 4
        ax.scatter(points[:, 0], points[:, 1], c=phases, cmap='hsv', s=node_sizes, zorder=2, edgecolors='white', linewidth=0.2, alpha=0.8)

        # --- 4. Отрисовка Квазичастиц ---
        if isinstance(stable_particles, (list, np.ndarray)) and len(stable_particles) > 0:
            for p_item in stable_particles:
                clump_nodes = p_item.get("node_indices")
                if clump_nodes is not None and len(clump_nodes) > 2:
                    clump_points = points[clump_nodes]
                    try:
                        hull = ConvexHull(clump_points)
                        for simplex in hull.simplices:
                            ax.plot(clump_points[simplex, 0], clump_points[simplex, 1],
                                    color='lightgray', linewidth=1.5, alpha=0.8, zorder=4, dashes=[8, 4]) # Ярче и толще
                    except Exception:
                        pass

                pos = p_item["position"]
                label = (f"ID:{p_item['track_id']} | Size: {p_item['size']}\n"
                         f"Mass: {p_item['mass']:.2f} | Charge: {p_item['average_charge']:.2f}\n"
                         f"Age: {p_item['age']}")

                # Динамическое размещение подписи
                x_data_min, x_data_max = points[:, 0].min(), points[:, 0].max()
                y_data_min, y_data_max = points[:, 1].min(), points[:, 1].max()
                x_offset = (x_data_max - x_data_min) * 0.02
                y_offset = (y_data_max - y_data_min) * 0.02
                ha = 'left'
                va = 'bottom'
                if pos[0] > x_data_min + (x_data_max - x_data_min) * 0.75: ha = 'right'; x_offset *= -1
                if pos[1] > y_data_min + (y_data_max - y_data_min) * 0.75: va = 'top'; y_offset *= -1

                ax.text(pos[0] + x_offset, pos[1] + y_offset, label, color='#00FFFF', fontsize=10,
                        fontweight='bold', ha=ha, va=va, zorder=5,
                        bbox=dict(facecolor='black', alpha=0.75, edgecolor='#00FFFF', boxstyle='round,pad=0.4'))

        # --- 5. Финальная настройка и сохранение ---
        title = (f"Project Genesis v10.2 | SEED: {shared_info['seed']}\n"
                 f"IC: {shared_info['ic']} | Frame: {frame_num} | "
                 f"Tracked: {tracked_count} | Stable: {len(stable_particles) if isinstance(stable_particles, (list, np.ndarray)) else 0}")
        ax.set_title(title, fontsize=14, color='white', pad=20)

        # --- АДАПТИВНЫЕ ПОЛЯ ---
        # Находим границы данных
        x_min_data, x_max_data = points[:, 0].min(), points[:, 0].max()
        y_min_data, y_max_data = points[:, 1].min(), points[:, 1].max()
        data_width = x_max_data - x_min_data
        data_height = y_max_data - y_min_data

        # Вычисляем центр данных
        center_x = (x_max_data + x_min_data) / 2
        center_y = (y_max_data + y_min_data) / 2

        # Определяем, какая сторона ограничивает (ширина или высота)
        aspect_ratio_data = data_width / data_height if data_height > 0 else 1
        aspect_ratio_fig = FIG_WIDTH_INCHES / FIG_HEIGHT_INCHES

        if aspect_ratio_data > aspect_ratio_fig:
            # Данные "шире" чем кадр, ширина определяет масштаб
            view_width = data_width * 1.1 # Добавляем 10% полей
            view_height = view_width / aspect_ratio_fig
        else:
            # Данные "выше" чем кадр, высота определяет масштаб
            view_height = data_height * 1.1 # Добавляем 10% полей
            view_width = view_height * aspect_ratio_fig

        ax.set_xlim(center_x - view_width / 2, center_x + view_width / 2)
        ax.set_ylim(center_y - view_height / 2, center_y + view_height / 2)

        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0)

        frame_filename = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
        fig.savefig(frame_filename, dpi=DPI, facecolor='black')
        plt.close(fig)

        return None

    except Exception as e:
        error_message = f"Frame {frame_num}: Error - {type(e).__name__} - {e}"
        cprint(f"ERROR rendering frame: {error_message}", "red")
        return error_message
