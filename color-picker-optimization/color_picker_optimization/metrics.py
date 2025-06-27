from skimage import color
import numpy as np

def cmyk_to_rgb(c, m, y, k):
    R = (1 - c) * (1 - k)
    G = (1 - m) * (1 - k)
    B = (1 - y) * (1 - k)
    max_val = max(R, G, B, 1)
    return tuple(np.clip([R / max_val, G / max_val, B / max_val], 0, 1))

def rgb2cmyk(rgb):
    """
    Convert an (H,W,3) uint8 RGB array (0–255) to an (H,W,4) float CMYK array (0–1).
    Output channels are ordered [C, M, Y, K].
    """
    # 1) Normalize RGB to [0,1]
    rgb_norm = rgb.astype(np.float32) / 255.0

    # 2) Split channels
    r = rgb_norm[..., 0]
    g = rgb_norm[..., 1]
    b = rgb_norm[..., 2]

    # 3) Compute K channel
    k = 1.0 - np.max(rgb_norm, axis=-1)

    # 4) Compute C, M, Y with protection against divide-by-zero
    denom = 1.0 - k
    c = np.where(denom > 0, (1.0 - r - k) / denom, 0.0)
    m = np.where(denom > 0, (1.0 - g - k) / denom, 0.0)
    y = np.where(denom > 0, (1.0 - b - k) / denom, 0.0)

    # 5) Stack into (H,W,4)
    return np.stack([c, m, y, k], axis=-1)
    
# Delta E (CIE76) from RGB
def delta_e_rgb(rgb1, rgb2):
    lab1 = color.rgb2lab(np.array(rgb1).reshape(1, 1, 3)).flatten()
    lab2 = color.rgb2lab(np.array(rgb2).reshape(1, 1, 3)).flatten()
    return np.linalg.norm(lab1 - lab2)

def calculate_error(colors, target_cmyk):
    target_rgb = cmyk_to_rgb(*target_cmyk)
    errors = []

    for color_cmyk in colors:
        rgb = cmyk_to_rgb(*color_cmyk)
        delta_e = delta_e_rgb(rgb, target_rgb)
        errors.append(delta_e)
        
    return errors

def calculate_plate_delta_e_array(plate_colors, target_cmyk):
    """
    Returns:
      errors: np.ndarray of shape (n_wells,)
              ΔE distances in the order of plate_colors.values()
    """
    target_rgb = cmyk_to_rgb(*target_cmyk)
    errors = [
        delta_e_rgb(rgb.astype(np.float32) / 255.0, target_rgb)
        for rgb in plate_colors.values()
    ]
    return np.array(errors, dtype=np.float32)
