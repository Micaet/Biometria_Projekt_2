import os
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

matplotlib.use('TkAgg')


def unwrap_iris(img, cx, cy, r_pupil, r_iris, size_out=(360, 60)):

    width, height = size_out
    theta = np.linspace(0, 2 * np.pi, width)
    r_vars = np.linspace(0, 1, height)


    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r_current = r_pupil + r_vars[i] * (r_iris - r_pupil)
            map_x[i, j] = cx + r_current * np.cos(theta[j])
            map_y[i, j] = cy + r_current * np.sin(theta[j])

    unwrapped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return unwrapped


def process_eye_projections_pro(image_path, xp_pupil=0.2):
    img = cv2.imread(image_path)
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p_mean = np.mean(gray)

    _, binary_pupil = cv2.threshold(gray, p_mean * xp_pupil, 255, cv2.THRESH_BINARY_INV)
    kernel_pupil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_pupil = cv2.morphologyEx(binary_pupil, cv2.MORPH_CLOSE, kernel_pupil)
    mask_pupil = cv2.morphologyEx(mask_pupil, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    proj_v_p = np.sum(mask_pupil, axis=0)
    proj_h_p = np.sum(mask_pupil, axis=1)

    if np.max(proj_v_p) == 0 or np.max(proj_h_p) == 0:
        return img, mask_pupil, np.zeros_like(gray), img, np.zeros((60, 360, 3), dtype=np.uint8)

    cx = np.argmax(proj_v_p)
    cy = np.argmax(proj_h_p)

    r_pupil = int((np.sum(mask_pupil[cy, :] == 255) + np.sum(mask_pupil[:, cx] == 255)) / 4)
    if r_pupil < 5: r_pupil = 20

    iris_gray = cv2.medianBlur(gray, 11)
    circles = cv2.HoughCircles(
        iris_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=50,
        param2=45,
        minRadius=int(r_pupil * 1.8),
        maxRadius=int(r_pupil * 3.5)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = circles[0, 0]
        r_iris = int(best_circle[2])
        dist = np.sqrt((best_circle[0] - cx) ** 2 + (best_circle[1] - cy) ** 2)
        if dist > r_pupil:
            r_iris = int(r_pupil * 2.8)
    else:
        r_iris = int(r_pupil * 2.8)

    unwrapped = unwrap_iris(img, cx, cy, r_pupil, r_iris)

    mask_iris = np.zeros_like(gray)
    cv2.circle(mask_iris, (cx, cy), r_iris, 255, -1)
    mask_iris = cv2.subtract(mask_iris, mask_pupil)

    res = img.copy()
    cv2.circle(res, (cx, cy), r_iris, (255, 0, 0), 2)
    cv2.circle(res, (cx, cy), r_pupil, (0, 255, 0), 2)
    cv2.drawMarker(res, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

    return img, mask_pupil, mask_iris, res, unwrapped


all_images = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            all_images.append(os.path.join(root, file))


if all_images:
    random_samples = random.sample(all_images, min(3, len(all_images)))
    plt.figure(figsize=(18, 12))

    for i, path in enumerate(random_samples):
        processed = process_eye_projections_pro(path)
        if processed is None: continue
        orig, m_p, m_i, det, flat = processed


        plt.subplot(3, 5, i * 5 + 1)
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title("Oryginał")
        plt.axis('off')


        plt.subplot(3, 5, i * 5 + 2)
        plt.imshow(m_p, cmap='gray')
        plt.title("Maska Źrenicy")
        plt.axis('off')

        plt.subplot(3, 5, i * 5 + 3)
        plt.imshow(m_i, cmap='gray')
        plt.title("Maska Tęczówki")
        plt.axis('off')


        plt.subplot(3, 5, i * 5 + 4)
        plt.imshow(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
        plt.title("Detekcja")
        plt.axis('off')

        # Rozwinięta tęczówka
        plt.subplot(3, 5, i * 5 + 5)
        plt.imshow(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB))
        plt.title("Rozwinięcie (Prostokąt)")

    plt.tight_layout()
    plt.show()