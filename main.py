
import os
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

def process_eye_projections_pro(image_path, xp=0.2):
    img = cv2.imread(image_path)
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    p_mean = np.mean(gray)
    _, binary = cv2.threshold(gray, p_mean * xp, 255, cv2.THRESH_BINARY_INV)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_clean)


    morphed = cv2.dilate(morphed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


    proj_v = np.sum(morphed, axis=0)
    proj_h = np.sum(morphed, axis=1)



    cx = np.argmax(proj_v)
    cy = np.argmax(proj_h)


    width_at_center = np.sum(morphed[cy, :] == 255)
    height_at_center = np.sum(morphed[:, cx] == 255)

    radius = int((width_at_center + height_at_center) / 4)

    if radius < 5: radius = 20


    res = img.copy()
    cv2.circle(res, (cx, cy), radius, (0, 255, 0), 2)
    cv2.drawMarker(res, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

    return img, morphed, res

all_images = []

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            all_images.append(os.path.join(root, file))


random_samples = random.sample(all_images, min(4, len(all_images)))

plt.figure(figsize=(12, 8))

for i, path in enumerate(random_samples):
    orig, mask, det = process_eye_projections_pro(path)

    if orig is None: continue


    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Oryginał: {os.path.basename(path)}")
    plt.axis('off')

    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Maska (Źrenica)")
    plt.axis('off')

    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
    plt.title("Środek (Projekcja)")
    plt.axis('off')

plt.tight_layout()
plt.show()