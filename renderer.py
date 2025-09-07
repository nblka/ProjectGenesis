# renderer.py v10.1
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
    v10.1:
    - ВОЗВРАЩЕНА правильная формула для node_sizes, чтобы размер узла зависел от |psi|^2.
    - Улучшен zorder для корректного наложения слоев.
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

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_facecolor('#0d0d0d')
        ax.set_aspect('equal')

        num_points = len(points)
        amplitudes_sq = np.abs(psi)**2

        # 1. Отрисовка 'ткани' пространства (треугольников)
        patches = []
        face_colors = []
        cmap_bg = plt.get_cmap('magma') # Можно использовать другой colormap для фона

        vmax = np.percentile(amplitudes_sq, 99.5) if num_points > 0 else 1.0
        vmin = np.min(amplitudes_sq)
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)

        if simplices is not None and simplices.ndim == 2 and simplices.shape[1] == 3:
            for simplex in simplices:
                if np.any(simplex >= num_points): continue
                patches.append(Polygon(points[simplex], closed=True))
                face_colors.append(cmap_bg(norm(np.mean(amplitudes_sq[simplex]))))

        collection = PatchCollection(patches, facecolors=face_colors, edgecolors='#333333', linewidth=0.5, alpha=0.9, zorder=1)
        ax.add_collection(collection)

        # 2. Отрисовка узлов (фаза - цвет, амплитуда - размер)
        phases = np.angle(psi)

        # <<< ИСПРАВЛЕНИЕ: ВОЗВРАЩАЕМ ПРАВИЛЬНУЮ ФОРМУЛУ ДЛЯ РАЗМЕРА УЗЛОВ >>>
        # Размер узла теперь сильно зависит от квадрата амплитуды |psi|^2
        # Это сделает "комки" визуально гораздо заметнее, а "вакуум" - почти невидимым.
        node_sizes = amplitudes_sq * 8000 + 4

        ax.scatter(points[:, 0], points[:, 1], c=phases, cmap='hsv', s=node_sizes, zorder=2, edgecolors='white', linewidth=0.2, alpha=0.8)

        # 3. Отрисовка найденных Квазичастиц
        if isinstance(stable_particles, (list, np.ndarray)) and len(stable_particles) > 0:
            for p_item in stable_particles:
                # ... (этот блок кода для подсветки границ и подписей остается без изменений) ...
                clump_nodes = p_item.get("node_indices")
                if clump_nodes is not None and len(clump_nodes) > 2:
                    clump_points = points[clump_nodes]
                    try:
                        hull = ConvexHull(clump_points)
                        for simplex in hull.simplices:
                            ax.plot(clump_points[simplex, 0], clump_points[simplex, 1],
                                    color='cyan', linewidth=2.5, alpha=0.7, zorder=3)
                    except Exception:
                        pass

                pos = p_item["position"]
                label = (f"ID:{p_item['track_id']} | Size: {p_item['size']}\n"
                         f"Mass: {p_item['mass']:.2f} | Charge: {p_item['average_charge']:.2f}\n"
                         f"Age: {p_item['age']}")

                # ... (динамическое размещение подписи) ...
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                x_offset = (x_max - x_min) * 0.02
                y_offset = (y_max - y_min) * 0.02
                ha = 'left'
                va = 'bottom'
                if pos[0] > (x_min + x_max) / 2: ha = 'right'; x_offset *= -1
                if pos[1] > (y_min + y_max) / 2: va = 'top'; y_offset *= -1

                ax.text(pos[0] + x_offset, pos[1] + y_offset, label, color='cyan', fontsize=9,
                        ha=ha, va=va, zorder=5,
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan', boxstyle='round,pad=0.3'))

        # Финальная настройка и сохранение
        title = (f"Project Genesis v10.1 | SEED: {shared_info['seed']}\n"
                 f"IC: {shared_info['ic']} | Frame: {frame_num} | " # Используем правильный номер кадра
                 f"Tracked: {tracked_count} | Stable: {len(stable_particles) if isinstance(stable_particles, (list, np.ndarray)) else 0}")
        ax.set_title(title, fontsize=12, color='white', pad=10)

        padding = 2.0
        x_min_data, x_max_data = points[:, 0].min(), points[:, 0].max()
        y_min_data, y_max_data = points[:, 1].min(), points[:, 1].max()
        ax.set_xlim(x_min_data - padding, x_max_data + padding)
        ax.set_ylim(y_min_data - padding, y_max_data + padding)

        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0.5)

        # Имя файла теперь тоже использует правильный номер кадра
        frame_filename = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
        fig.savefig(frame_filename, dpi=150, facecolor='black')
        plt.close(fig)

        return None

    except Exception as e:
        error_message = f"Frame {frame_num}: Error - {type(e).__name__} - {e}"
        cprint(f"ERROR rendering frame: {error_message}", "red")
        return error_message
