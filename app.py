import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils import process_eye_projections_pro, unwrap_iris

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

        self.label_left = tk.Label(root, text="Lewe oko: brak")
        self.label_left.pack()

        self.label_right = tk.Label(root, text="Prawe oko: brak")
        self.label_right.pack()

        self.frame_main = tk.Frame(root)
        self.frame_main.pack()

        self.left_col = tk.Frame(self.frame_main)
        self.left_col.grid(row=0, column=0, padx=10)

        self.right_col = tk.Frame(self.frame_main)
        self.right_col.grid(row=0, column=1, padx=10)
        self.root.geometry("1000x900")
        self.labels = []

    def load_image(self, side):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        filename = path.split("/")[-1].split("\\")[-1]  # tylko nazwa pliku

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

        self.display_pipeline(
            self.img1_path,
            self.left_col,
            row=0,
            title="LEWE OKO"
        )

        self.display_pipeline(
            self.img2_path,
            self.left_col,
            row=2,
            title="PRAWE OKO"
        )

    def display_pipeline(self, path, parent, row=0, title=""):
        result = process_eye_projections_pro(path)
        if result is None:
            return

        orig, m_p, m_i, det, flat = result

        code = BiometriaApp.iris_code(flat)
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
    def encode_band(band, convolve_maker = "scipy"):
        if len(band.shape) == 3:
            band = BiometriaApp.greyscale(band)

        band = band.astype(np.float32)

        band = (band - np.mean(band)) / (np.std(band) + 1e-5)

        freq = 0.1 # + próg hamminga
        sigma = 0.5 * np.pi * freq
        ksize = 9

        k_real, k_imag = BiometriaApp.gabor_kernel(ksize, sigma, freq)

        real = BiometriaApp.convolve_type(band, k_real, convolve_maker)
        imag = BiometriaApp.convolve_type(band, k_imag, convolve_maker)

        code_real = real > 0
        code_imag = imag > 0

        code = np.stack([code_real, code_imag], axis=-1)

        return code.reshape(-1)

    @staticmethod
    def gabor_kernel(ksize, sigma, freq):
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
    def iris_code(flat):
        h, w = flat.shape[:2]
        bands = 8
        bh = h // bands

        full_code = []

        for i in range(bands):
            band = flat[i * bh:(i + 1) * bh, :]

            band = band[2:-2, :]

            code = BiometriaApp.encode_band(band)
            full_code.append(code)

        return np.concatenate(full_code)

    @staticmethod
    def compare_iris(flat1, flat2):
        code1 = BiometriaApp.iris_code(flat1)
        code2 = BiometriaApp.iris_code(flat2)

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
            out[y - 1:y + 1, :] = 255  # linia oddzielająca

        return out


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()