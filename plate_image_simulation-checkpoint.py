import os
import numpy as np
import cv2
from typing import List, Tuple, Optional


def cmyk_to_bgr(cmyk: List[float]) -> Tuple[int, int, int]:
    """
    Convert a CMYK color (0-1 floats) into a BGR tuple (0-255 ints) for OpenCV.
    """
    c, m, y, k = cmyk
    # undercolor removal
    r = 1.0 - min(1.0, c * (1 - k) + k)
    g = 1.0 - min(1.0, m * (1 - k) + k)
    b = 1.0 - min(1.0, y * (1 - k) + k)
    return (int(255 * b), int(255 * g), int(255 * r))

def simulate_mixing(volume_sets):
    results = []
    for volumes in volume_sets:
        total = sum(volumes)
        if total == 0:
            results.append([0.0, 0.0, 0.0, 0.0])
        else:
            normalized = [round(v / total, 3) for v in volumes]
            results.append(normalized)
    return results


def simulate_plate_image(
    cmyk_colors: List[List[float]],
    output_path: str = 'simulated_plate_image.jpg',
    show: bool = False
) -> np.ndarray:
    """
    Load well positions and template from the bundled example_data .npz,
    fill wells with provided CMYK colors, and save a simulated plate image.

    Args:
        cmyk_colors: List of [C, M, Y, K] floats in 0-1. Mapped sequentially to wells A1, A2, ...
        output_path:   Filename where the simulated image will be saved (in working dir).
        show:          If True, display the result via matplotlib.

    Returns:
        The simulated BGR image as a NumPy array.
    """
    # determine path to our example_data folder
    module_dir = os.path.dirname(__file__)
    npz_path = os.path.join(module_dir, '..', 'example_data', 'image_simulation_reqs.npz')
    npz_path = os.path.normpath(npz_path)

    # load precomputed data
    data = np.load(npz_path, allow_pickle=True)
    well_points: Dict[str, np.ndarray] = data['well_points'].item()
    wells_img:     np.ndarray = data['wells_img']
    plateM:        np.ndarray = data['plateM']

    # initialize all wells as white (CMYK = 0,0,0,0)
    color_dict = {name: [0.0, 0.0, 0.0, 0.0] for name in well_points}
    well_names = list(well_points.keys())

    # assign sequential colors
    for i, color in enumerate(cmyk_colors):
        if i >= len(well_names):
            break
        color_dict[well_names[i]] = color

    # create the simulated image
    simulated = wells_img.copy()
    radius = int(plateM[0, 0] / 2)

    for name, (x, y) in well_points.items():
        bgr = cmyk_to_bgr(color_dict[name])
        cv2.circle(simulated, (int(x), int(y)), radius, bgr, -1)

    # save to disk
    cv2.imwrite(output_path, simulated)

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(simulated, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return simulated
