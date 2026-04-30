import cv2
import numpy as np


def unwrap_iris(img, cx, cy, r_pupil, r_iris, size_out=(360, 120)):
    width, height = size_out
    theta_right = np.linspace(np.radians(-30), np.radians(80), width // 2)
    theta_left = np.linspace(np.radians(100), np.radians(230), width // 2)
    theta = np.concatenate([theta_right, theta_left])
    r_vars = np.linspace(0.05, 0.85, height)

    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r_current = r_pupil + r_vars[i] * (r_iris - r_pupil)
            map_x[i, j] = cx + r_current * np.cos(theta[j])
            map_y[i, j] = cy + r_current * np.sin(theta[j])

    unwrapped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    if len(unwrapped.shape) == 3:
        unwrapped = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    unwrapped = clahe.apply(unwrapped.astype(np.uint8))
    return unwrapped


def process_eye_projections_pro(image_path, xp_pupil=0.2):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p_mean = np.mean(gray)
    _, binary_pupil = cv2.threshold(gray, p_mean * xp_pupil, 255, cv2.THRESH_BINARY_INV)
    kernel_pupil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_close = cv2.morphologyEx(binary_pupil, cv2.MORPH_CLOSE, kernel_pupil)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    proj_v_p = np.sum(mask_open, axis=0)
    proj_h_p = np.sum(mask_open, axis=1)
    if np.max(proj_v_p) == 0 or np.max(proj_h_p) == 0:
        return img, mask_open, np.zeros_like(gray), img, np.zeros((120, 360), dtype=np.uint8)

    cx, cy = np.argmax(proj_v_p), np.argmax(proj_h_p)
    r_pupil = int((np.sum(mask_open[cy, :] == 255) + np.sum(mask_open[:, cx] == 255)) / 4)
    if r_pupil < 5: r_pupil = 20

    iris_gray = cv2.medianBlur(gray, 11)
    circles = cv2.HoughCircles(iris_gray, cv2.HOUGH_GRADIENT, 1.2, 200, param1=50, param2=45,
                               minRadius=int(r_pupil * 1.8), maxRadius=int(r_pupil * 3.5))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        r_iris = int(circles[0, 0][2])
    else:
        r_iris = int(r_pupil * 2.5)

    unwrapped = unwrap_iris(img, cx, cy, r_pupil, r_iris)
    res_circles = img.copy()
    cv2.circle(res_circles, (cx, cy), r_iris, (255, 0, 0), 2)
    cv2.circle(res_circles, (cx, cy), r_pupil, (0, 255, 0), 2)
    return img, mask_open, iris_gray, res_circles, unwrapped


def get_diagnostic_steps(image_path, xp_pupil=0.2):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p_mean = np.mean(gray)
    _, binary = cv2.threshold(gray, p_mean * xp_pupil, 255, cv2.THRESH_BINARY_INV)
    k_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_p)
    m_open = cv2.morphologyEx(m_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    res = process_eye_projections_pro(image_path, xp_pupil)
    if not res: return None
    _, _, hough_in, circ_img, unwrapped = res

    return {
        "1. Szarość": gray,
        "2. Binarizacja": binary,
        "3. Morph Close": m_close,
        "4. Morph Open": m_open,
        "5. Hough Input": hough_in,
        "6. Okręgi": circ_img,
        "7. Unwrap": unwrapped
    }