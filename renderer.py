# renderer.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

def render_frame_worker(args_tuple):
    frame_idx, data_path, frames_dir, shared_info = args_tuple

    try:
        def unwrap(d):
            if hasattr(d, 'ndim') and d.ndim == 0 and hasattr(d, 'dtype') and d.dtype == 'object':
                return d.item()
            return d

        data = np.load(data_path, allow_pickle=True)
        points = unwrap(data['points'])
        psi = unwrap(data['psi'])
        simplices = unwrap(data['simplices'])
        stable_particles = unwrap(data['stable_particles'])
        tracked_count = unwrap(data['tracked_count'])

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#0d0d0d')

        num_points = len(points)

        # --- Отрисовка 'ткани' пространства (треугольников) ---
        bg_colors_per_node = np.abs(psi)**2
        patches = []
        face_colors = []
        cmap = plt.get_cmap('inferno')
        vmax = np.percentile(bg_colors_per_node, 99.5) if len(bg_colors_per_node) > 0 else 1.0
        vmin = np.min(bg_colors_per_node)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if simplices is not None and simplices.size > 0 and simplices.ndim == 2:
            for simplex in simplices:
                if np.any(simplex >= num_points): continue
                patches.append(Polygon(points[simplex], closed=True))
                face_colors.append(cmap(norm(np.mean(bg_colors_per_node[simplex]))))
        collection = PatchCollection(patches, facecolors=face_colors, edgecolors='#444444', linewidth=0.4, alpha=0.8)
        ax.add_collection(collection)

        # --- Отрисовка узлов (фаза - цвет, амплитуда - размер) ---
        phases = np.angle(psi)
        amplitudes = np.abs(psi)
        node_sizes = (amplitudes * 150)**1.8 + 5
        ax.scatter(points[:, 0], points[:, 1], c=phases, cmap='hsv', s=node_sizes, zorder=2, edgecolors='white', linewidth=0.2, alpha=0.9)

        if isinstance(stable_particles, (list, np.ndarray)) and len(stable_particles) > 0:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            for p_item in stable_particles:
                pos = p_item["position"]
                track_id = p_item["track_id"]
                label = f"ID:{track_id}\nMass: {p_item['mass']:.2f}\nCharge: {p_item['average_charge']:.2f}\nAge: {p_item['age']}"

                ax.scatter(pos[0], pos[1], s=300, facecolors='none', edgecolors='cyan', linewidths=2, zorder=3)

                # 1. Задаем смещения и выравнивание по умолчанию (вверх-вправо)
                x_offset = 1.5 # Увеличим отступ для красоты
                y_offset = 1.5
                ha = 'left'  # horizontalalignment
                va = 'bottom' # verticalalignment

                # 2. Проверяем, не слишком ли близко к правому краю
                if pos[0] > x_max - x_range * 0.2: # Если в правых 20% экрана
                    x_offset = -1.5
                    ha = 'right'

                # 3. Проверяем, не слишком ли близко к верхнему краю
                if pos[1] > y_max - y_range * 0.2: # Если в верхних 20% экрана
                    y_offset = -1.5
                    va = 'top'

                ax.text(pos[0] + x_offset, pos[1] + y_offset, label, color='cyan', fontsize=9,
                        ha=ha, va=va, # Используем динамическое выравнивание
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='cyan', boxstyle='round,pad=0.4'))

        # --- Настройка и сохранение кадра ---
        title = (
            f"Project Genesis v9.0 'Crystal' | SEED: {shared_info['seed']} | Topology: {shared_info['topology']}\n"
            f"IC: {shared_info['ic']} | Frame: {frame_idx+1} | Tracked: {tracked_count} | Stable: {len(stable_particles) if isinstance(stable_particles, (list, np.ndarray)) else 0}"
        )
        ax.set_title(title, fontsize=12, color='white', pad=10)

        ax.set_aspect('equal')

        # Устанавливаем границы видимой области на основе реальных
        # координат точек, а не абстрактной ширины/высоты.
        padding = 2.0 # Небольшой отступ по краям
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        # ### КОНЕЦ ИЗМЕНЕНИЙ ###

        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0.5)

        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        fig.savefig(frame_filename, dpi=150, facecolor='black')
        plt.close(fig)

        return None

    except Exception as e:
        error_message = f"Frame {frame_idx+1}: Error - {type(e).__name__} - {e}"
        print(error_message)
        return error_message
