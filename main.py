import os
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2
from utils import process_eye_projections_pro, unwrap_iris

matplotlib.use('TkAgg')





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