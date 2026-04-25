import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils import process_eye_projections_pro, unwrap_iris
import os
import csv

from scipy.ndimage import convolve



class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie tęczówek")

        self.img1_path = None
        self.img2_path = None
        self.convolve_maker = "scipy"

        self.frame_top = tk.Frame(root)
        self.frame_top.pack()

        self.btn_load1 = tk.Button(
            self.frame_top,
            text="Wczytaj lewe oko",
            command=lambda: self.load_image("left")
        )
        self.btn_load1.grid(row=0, column=0, padx=5)

        self.btn_load2 = tk.Button(
            self.frame_top,
            text="Wczytaj prawe oko",
            command=lambda: self.load_image("right")
        )
        self.btn_load2.grid(row=0, column=1, padx=5)

        self.btn_process = tk.Button(
            self.frame_top,
            text="Przetwórz",
            command=self.process_images
        )
        self.btn_process.grid(row=0, column=2, padx=5)

        tk.Label(self.frame_top, text="Próg:").grid(row=0, column=3, padx=(15, 2))
        self.threshold_var = tk.DoubleVar(value=0.4009) # taki wyszedl optymalny
        self.threshold_entry = tk.Entry(
            self.frame_top, textvariable=self.threshold_var, width=6
        )
        self.threshold_entry.grid(row=0, column=4, padx=5)

        self.label_left = tk.Label(root, text="Lewe oko: brak")
        self.label_left.pack()

        self.label_right = tk.Label(root, text="Prawe oko: brak")
        self.label_right.pack()

        self.frame_main = tk.Frame(root)
        self.frame_main.pack()

        self.left_col = tk.Frame(self.frame_main)
        self.left_col.grid(row=0, column=0, padx=10)

        self.right_col = tk.Frame(self.frame_main)
        self.frame_result = tk.Frame(root, relief=tk.GROOVE, bd=2)
        self.frame_result.pack(fill=tk.X, padx=10, pady=10)

        self.root.geometry("1000x1000")
        self.labels = []

        self.flat_left = None
        self.flat_right = None

        self.used_freq = 0.2892
        self.used_treshold = 0.4009

    def load_image(self, side):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        filename = path.split("/")[-1].split("\\")[-1]

        if side == "left":
            self.img1_path = path
            self.label_left.config(text=f"Lewe: {filename}")
            print("Lewe:", filename)

        elif side == "right":
            self.img2_path = path
            self.label_right.config(text=f"Prawe: {filename}")
            print("Prawe:", filename)

    def process_images(self):
        if not self.img1_path or not self.img2_path:
            print("Wgraj oba obrazy")
            return

        for widget in self.left_col.winfo_children():
            widget.destroy()

        flat_left = self.display_pipeline(self.img1_path, self.left_col, row=0, title="LEWE OKO" )

        flat_right = self.display_pipeline(self.img2_path, self.left_col, row=2, title="PRAWE OKO")

        self.flat_left = flat_left
        self.flat_right = flat_right

        if flat_left is not None and flat_right is not None:
            threshold = self.threshold_var.get()
            self.check_iris(flat_left, flat_right, threshold)

    def display_pipeline(self, path, parent, row=0, title=""):
        result = process_eye_projections_pro(path)
        if result is None:
            return None

        orig, m_p, m_i, det, flat = result

        code = BiometriaApp.iris_code(flat, self.used_freq, self.convolve_maker)
        code_img = BiometriaApp.code_to_image(code)

        images = [
            ("Oryginał", orig, True),
            ("Detekcja", det, True),
            ("Unwrap", flat, True),
            ("Kod tęczówki", code_img, False),
        ]

        tk.Label(parent, text=title).grid(row=row, column=0, columnspan=4)

        for col, (name, img, is_color) in enumerate(images):
            frame = tk.Frame(parent)
            frame.grid(row=row + 1, column=col, padx=5)
            tk.Label(frame, text=name).pack()
            img_disp = self.prepare_image(img, is_color)
            lbl = tk.Label(frame, image=img_disp)
            lbl.image = img_disp
            lbl.pack()

        return flat


    def check_iris(self, flat_left, flat_right, threshold=0.4009):
        for w in self.frame_result.winfo_children():
            w.destroy()

        db = self._load_code_database()
        if not db:
            tk.Label(
                self.frame_result,
                text=f"Brak bazy kodów w 'iris_codes/', uruchom create_iris_records.py",
                fg="red"
            ).pack()
            return

        query_left = BiometriaApp.iris_code(flat_left, self.used_freq, self.convolve_maker)
        query_right = BiometriaApp.iris_code(flat_right, self.used_freq, self.convolve_maker)

        best_person = None
        best_hd = float("inf")
        best_hd_l = None
        best_hd_r = None

        persons = set(pid for pid, _ in db.keys())

        for pid in persons:
            hd_l = hd_r = None

            if (pid, "left") in db:
                hd_l = BiometriaApp.hamming_distance(query_left, db[(pid, "left")])
            if (pid, "right") in db:
                hd_r = BiometriaApp.hamming_distance(query_right, db[(pid, "right")])

            if hd_l is not None and hd_r is not None:
                hd_avg = (hd_l + hd_r) / 2
            elif hd_l is not None:
                hd_avg = hd_l
            else:
                hd_avg = hd_r

            if hd_avg < best_hd:
                best_hd = hd_avg
                best_person = pid
                best_hd_l = hd_l
                best_hd_r = hd_r

        match = best_hd <= threshold
        color = "green" if match else "red"
        verdict = "DOPASOWANO" if match else "BRAK DOPASOWANIA"

        tk.Label(
            self.frame_result,
            text=f"Wynik: {verdict}  |  Osoba: {best_person}  |  "
                 f"HD śr.: {best_hd:.4f}  (L: {best_hd_l:.4f}, P: {best_hd_r:.4f})  |  "
                 f"Próg: {threshold}",
            fg=color,
            font=("Helvetica", 11, "bold")
        ).pack(pady=4)

        img_paths = self._find_person_images(best_person)
        if img_paths:
            eyes_frame = tk.Frame(self.frame_result)
            eyes_frame.pack(pady=4)
            tk.Label(eyes_frame, text=f"Oczy osoby {best_person} z bazy:").grid(
                row=0, column=0, columnspan=2
            )
            for col, (side_label, img_path) in enumerate(img_paths):
                if img_path is None:
                    continue
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                img_disp = self._cv_to_tk(img_cv, max_w=250, max_h=200)
                sub = tk.Frame(eyes_frame)
                sub.grid(row=1, column=col, padx=10)
                tk.Label(sub, text=side_label).pack()
                lbl = tk.Label(sub, image=img_disp)
                lbl.image = img_disp
                lbl.pack()

    def _load_code_database(self):
        manifest = os.path.join("iris_codes", "manifest.csv")
        if not os.path.isfile(manifest):
            return {}

        db = {}
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code_path = row["code_file"]
                if os.path.isfile(code_path):
                    db[(row["person_id"], row["side"])] = np.load(code_path)
        return db

    def _find_person_images(self, person_id):
        manifest = os.path.join("iris_codes", "manifest.csv")
        if not os.path.isfile(manifest):
            return []

        result = {}
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["person_id"] == person_id:
                    result[row["side"]] = row["source_file"]

        pairs = []
        for side in ("left", "right"):
            label = "Lewe oko" if side == "left" else "Prawe oko"
            pairs.append((label, result.get(side)))
        return pairs

    @staticmethod
    def _cv_to_tk(img_cv, max_w=250, max_h=200):
        h, w = img_cv.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(img_rgb))

    def prepare_image(self, img, is_color):
        if is_color:
            img = BiometriaApp.greyscale(img)

        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255)
            img = img.astype(np.uint8)

        img = cv2.resize(img, (200, 150), interpolation=cv2.INTER_NEAREST)
        pil_img = Image.fromarray(img)
        return ImageTk.PhotoImage(pil_img)

    @staticmethod
    def code_to_image(code):
        length = len(code)
        width = length // 8
        height = 8
        code_trimmed = code[:height * width]
        img_2d = code_trimmed.reshape((height, width))
        return img_2d.astype(np.uint8) * 255

    @staticmethod
    def resize_bilinear(img, new_w, new_h):
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        if channels == 1:
            result = np.zeros((new_h, new_w), dtype=np.float32)
        else:
            result = np.zeros((new_h, new_w, channels), dtype=np.float32)

        x_ratio = (w - 1) / new_w
        y_ratio = (h - 1) / new_h

        for i in range(new_h):
            for j in range(new_w):
                x = x_ratio * j
                y = y_ratio * i

                x_l = int(x)
                y_l = int(y)
                x_h = min(x_l + 1, w - 1)
                y_h = min(y_l + 1, h - 1)

                x_weight = x - x_l
                y_weight = y - y_l

                a = img[y_l, x_l]
                b = img[y_l, x_h]
                c = img[y_h, x_l]
                d = img[y_h, x_h]

                pixel = (
                        a * (1 - x_weight) * (1 - y_weight) +
                        b * x_weight * (1 - y_weight) +
                        c * (1 - x_weight) * y_weight +
                        d * x_weight * y_weight
                )

                result[i, j] = pixel
        return result.astype(img.dtype)

    @staticmethod
    def encode_band(band, freq = 0.2892, convolve_maker = "scipy"): # częstotliwość wyznaczona eksperymentalnie
        if len(band.shape) == 3:
            band = BiometriaApp.greyscale(band)

        band = band.astype(np.float32)

        band = (band - np.mean(band)) / (np.std(band) + 1e-5)
        sigma = 0.5 * np.pi * freq
        ksize = 9

        k_real, k_imag = BiometriaApp.gabor_kernel(ksize, freq, sigma)

        real = BiometriaApp.convolve_type(band, k_real, convolve_maker)
        imag = BiometriaApp.convolve_type(band, k_imag, convolve_maker)

        code_real = real > 0
        code_imag = imag > 0

        code = np.stack([code_real, code_imag], axis=-1)

        return code.reshape(-1)

    @staticmethod
    def gabor_kernel(ksize, freq, sigma):
        real = np.zeros((ksize, ksize))
        imag = np.zeros((ksize, ksize))

        half = ksize // 2

        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                gauss = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                phase = 2 * np.pi * freq * x

                real[y + half, x + half] = gauss * np.cos(phase)
                imag[y + half, x + half] = gauss * np.sin(phase)

        return real, imag

    @staticmethod
    def greyscale(img):
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        grey = 0.114 * b + 0.587 * g + 0.299 * r
        return grey.astype(np.float32)

    @staticmethod
    def iris_code(flat, freq, convolve_maker):
        h, w = flat.shape[:2]
        bands = 8
        bh = h // bands

        full_code = []

        for i in range(bands):
            band = flat[i * bh:(i + 1) * bh, :]

            band = band[2:-2, :] # usuwamy czesc oka, by zmruzenie oka miało mniejszy efekt

            code = BiometriaApp.encode_band(band, freq, convolve_maker)
            full_code.append(code)

        return np.concatenate(full_code)

    @staticmethod
    def compare_iris(flat1, flat2, freq = 0.2892, convolve_maker = "scipy"):
        code1 = BiometriaApp.iris_code(flat1, freq, convolve_maker)
        code2 = BiometriaApp.iris_code(flat2, freq, convolve_maker)

        hd = BiometriaApp.hamming_distance(code1, code2)

        return hd

    @staticmethod
    def hamming_distance(c1, c2):
        min_len = min(len(c1), len(c2))
        return np.sum(c1[:min_len] != c2[:min_len]) / min_len

    @staticmethod
    def reflect_pad(matrix, pad_h, pad_w):
        m_h, m_w = matrix.shape
        new_h = m_h + 2 * pad_h
        new_w = m_w + 2 * pad_w
        padded = np.zeros((new_h, new_w))

        for i in range(m_h):
            for j in range(m_w):
                padded[i + pad_h][j + pad_w] = matrix[i][j]

        for i in range(pad_h):
            for j in range(m_w):
                padded[pad_h - 1 - i][j + pad_w] = matrix[i][j]
                padded[pad_h + m_h + i][j + pad_w] = matrix[m_h - 1 - i][j]

        for i in range(new_h):
            for j in range(pad_w):
                padded[i][pad_w - 1 - j] = padded[i][pad_w + j]
                padded[i][pad_w + m_w + j] = padded[i][pad_w + m_w - 1 - j]

        return padded

    @staticmethod
    def convolve_type(matrix, kernel, convolve_maker):
        if convolve_maker == "scipy":
            return convolve(matrix, kernel)
        else:
            kernel = kernel[::-1, ::-1]

            m_h, m_w = matrix.shape
            k_h, k_w = kernel.shape
            pad_h, pad_w = k_h // 2, k_w // 2

            padded = BiometriaApp.reflect_pad(matrix, pad_h, pad_w)

            output = np.zeros((m_h, m_w))
            for i in range(m_h):
                for j in range(m_w):
                    total = 0.0
                    for ki in range(k_h):
                        for kj in range(k_w):
                            total += padded[i + ki][j + kj] * kernel[ki][kj]
                    output[i][j] = total

            return output

    @staticmethod
    def show_bands(flat):
        h, w = flat.shape[:2]
        bands = 8
        bh = h // bands

        out = flat.copy()

        for i in range(1, bands):
            y = i * bh
            out[y - 1:y + 1, :] = 255

        return out


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()