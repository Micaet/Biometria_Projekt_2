import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from utils import process_eye_projections_pro, unwrap_iris, get_diagnostic_steps
import os
import csv
from scipy.ndimage import convolve


class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie tęczówek")
        self.root.geometry("1200x950")

        self.img1_path = None
        self.img2_path = None
        self.convolve_maker = "scipy"
        self.used_freq = 0.2892

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        self.tab_main = tk.Frame(self.notebook)
        self.tab_diag = tk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Panel Główny")
        self.notebook.add(self.tab_diag, text="Diagnostyka")

        self._setup_main_tab()
        self._setup_diag_tab()

    def _setup_main_tab(self):
        self.frame_top = tk.Frame(self.tab_main)
        self.frame_top.pack(pady=10)

        self.btn_load1 = tk.Button(self.frame_top, text="Wczytaj lewe oko", command=lambda: self.load_image("left"))
        self.btn_load1.grid(row=0, column=0, padx=5)

        self.btn_load2 = tk.Button(self.frame_top, text="Wczytaj prawe oko", command=lambda: self.load_image("right"))
        self.btn_load2.grid(row=0, column=1, padx=5)

        self.btn_process = tk.Button(self.frame_top, text="Przetwórz", command=self.process_images)
        self.btn_process.grid(row=0, column=2, padx=5)

        tk.Label(self.frame_top, text="Próg:").grid(row=0, column=3, padx=(15, 2))
        self.threshold_var = tk.DoubleVar(value=0.401)
        self.threshold_entry = tk.Entry(self.frame_top, textvariable=self.threshold_var, width=6)
        self.threshold_entry.grid(row=0, column=4, padx=5)

        self.label_left = tk.Label(self.tab_main, text="Lewe oko: brak");
        self.label_left.pack()
        self.label_right = tk.Label(self.tab_main, text="Prawe oko: brak");
        self.label_right.pack()

        self.frame_main = tk.Frame(self.tab_main);
        self.frame_main.pack()
        self.left_col = tk.Frame(self.frame_main);
        self.left_col.grid(row=0, column=0, padx=10)
        self.frame_result = tk.Frame(self.tab_main, relief=tk.GROOVE, bd=2);
        self.frame_result.pack(fill=tk.X, padx=10, pady=10)

    def _setup_diag_tab(self):
        self.diag_canvas = tk.Canvas(self.tab_diag)
        self.diag_scroll = ttk.Scrollbar(self.tab_diag, orient="vertical", command=self.diag_canvas.yview)
        self.diag_cont = tk.Frame(self.diag_canvas)
        self.diag_canvas.create_window((0, 0), window=self.diag_cont, anchor="nw")
        self.diag_canvas.configure(yscrollcommand=self.diag_scroll.set)
        self.diag_canvas.pack(side="left", fill="both", expand=True)
        self.diag_scroll.pack(side="right", fill="y")
        self.diag_cont.bind("<Configure>",
                            lambda e: self.diag_canvas.configure(scrollregion=self.diag_canvas.bbox("all")))

    def load_image(self, side):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        filename = os.path.basename(path)
        if side == "left":
            self.img1_path = path;
            self.label_left.config(text=f"Lewe: {filename}")
        else:
            self.img2_path = path;
            self.label_right.config(text=f"Prawe: {filename}")

    def process_images(self):
        if not self.img1_path or not self.img2_path: return
        for widget in self.left_col.winfo_children(): widget.destroy()

        flat_left = self.display_pipeline(self.img1_path, self.left_col, row=0, title="LEWE OKO")
        flat_right = self.display_pipeline(self.img2_path, self.left_col, row=2, title="PRAWE OKO")

        if flat_left is not None and flat_right is not None:
            self.check_iris(flat_left, flat_right, self.threshold_var.get())

        self.display_diagnostics_all()

    def display_pipeline(self, path, parent, row=0, title=""):
        result = process_eye_projections_pro(path)
        if result is None: return None
        orig, mask, hough, det, flat = result
        code = BiometriaApp.iris_code(flat, self.used_freq, self.convolve_maker)
        code_img = BiometriaApp.code_to_image(code)

        images = [("Oryginał", orig, True), ("Detekcja", det, True), ("Unwrap", flat, True), ("Kod", code_img, False)]
        tk.Label(parent, text=title, font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=4, pady=5)
        for col, (name, img, is_color) in enumerate(images):
            frame = tk.Frame(parent);
            frame.grid(row=row + 1, column=col, padx=5)
            tk.Label(frame, text=name).pack()
            img_disp = self.prepare_image(img, is_color)
            lbl = tk.Label(frame, image=img_disp);
            lbl.image = img_disp;
            lbl.pack()
        return flat

    def display_diagnostics_all(self):
        for w in self.diag_cont.winfo_children(): w.destroy()
        for i, path in enumerate([self.img1_path, self.img2_path]):
            if not path: continue
            steps = get_diagnostic_steps(path)
            if not steps: continue

            f_row = tk.LabelFrame(self.diag_cont, text="DIAGNOSTYKA - " + ("LEWE" if i == 0 else "PRAWE"),
                                  font=("Arial", 11, "bold"))
            f_row.pack(fill="x", padx=10, pady=10)

            flat = steps["7. Unwrap"]
            cropped = flat[5:flat.shape[0] - 5, :]
            code = BiometriaApp.iris_code(flat, self.used_freq, self.convolve_maker)

            steps["8. Po wycięciu"] = cropped
            steps["9. Kod finalny"] = BiometriaApp.code_to_image(code)

            for col, (name, img_data) in enumerate(steps.items()):
                f = tk.Frame(f_row);
                f.grid(row=0, column=col, padx=5, pady=5)
                tk.Label(f, text=name, font=("Arial", 8)).pack()
                sz = (160, 120) if "wycięciu" not in name else (160, 50)
                img_disp = self.prepare_image(img_data, False, size=sz)
                lbl = tk.Label(f, image=img_disp);
                lbl.image = img_disp;
                lbl.pack()

    def check_iris(self, flat_left, flat_right, threshold=0.401):
        for w in self.frame_result.winfo_children(): w.destroy()
        db = self._load_code_database()
        if not db: return

        query_left = BiometriaApp.iris_code(flat_left, self.used_freq, self.convolve_maker)
        query_right = BiometriaApp.iris_code(flat_right, self.used_freq, self.convolve_maker)

        best_person, best_hd = None, float("inf")
        for pid in set(p for p, _ in db.keys()):
            h_l = BiometriaApp.hamming_distance(query_left, db[(pid, "left")]) if (pid, "left") in db else 1.0
            h_r = BiometriaApp.hamming_distance(query_right, db[(pid, "right")]) if (pid, "right") in db else 1.0
            cur_min = min(h_l, h_r)
            if cur_min < best_hd: best_hd, best_person = cur_min, pid

        verdict = "DOPASOWANO" if best_hd <= threshold else "BRAK DOPASOWANIA"
        color = "green" if best_hd <= threshold else "red"

        tk.Label(
            self.frame_result,
            text=f"WYNIK: {verdict}\n"
                 f"Osoba: {best_person}  |  Wybrany próg w UI: {threshold:.4f}\n"
                 f"Najmniejszy próg do akceptacji tego zdjęcia: {best_hd:.4f}\n",
            fg=color, font=("Helvetica", 11, "bold"), justify=tk.CENTER
        ).pack(pady=10)

    @staticmethod
    def greyscale(img):
        if len(img.shape) == 3:
            return (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.float32)
        return img.astype(np.float32)

    def prepare_image(self, img, is_color, size=(200, 150)):
        if is_color and len(img.shape) == 3: img = BiometriaApp.greyscale(img)
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-5) * 255).astype(np.uint8)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(img))

    @staticmethod
    def iris_code(flat, freq, convolve_maker):
        h, w = flat.shape[:2]
        crop = 5
        flat = flat[crop:h - crop, :]
        h = flat.shape[0]
        bands, bh = 8, h // 8
        full_code = []
        for i in range(bands):
            band = flat[i * bh:(i + 1) * bh, :]
            code = BiometriaApp.encode_band(band, freq, convolve_maker)
            full_code.append(code)
        return np.concatenate(full_code)

    @staticmethod
    def encode_band(band, freq, convolve_maker):
        band = BiometriaApp.greyscale(band)
        band = (band - np.mean(band)) / (np.std(band) + 1e-5)
        sigma = 0.5 * np.pi * freq
        k_real, k_imag = BiometriaApp.gabor_kernel(5, freq, sigma)
        real = BiometriaApp.convolve_type(band, k_real, convolve_maker)
        imag = BiometriaApp.convolve_type(band, k_imag, convolve_maker)
        return np.stack([real > 0, imag > 0], axis=-1).reshape(-1)

    @staticmethod
    def gabor_kernel(ksize, freq, sigma):
        real, imag = np.zeros((ksize, ksize)), np.zeros((ksize, ksize))
        half = ksize // 2
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                gauss = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                phase = 2 * np.pi * freq * x
                real[y + half, x + half], imag[y + half, x + half] = gauss * np.cos(phase), gauss * np.sin(phase)
        return real, imag

    @staticmethod
    def convolve_type(matrix, kernel, convolve_maker):
        if convolve_maker == "scipy": return convolve(matrix, kernel)
        kernel = kernel[::-1, ::-1]
        m_h, m_w = matrix.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = BiometriaApp.reflect_pad(matrix, pad_h, pad_w)
        output = np.zeros((m_h, m_w))
        for i in range(m_h):
            for j in range(m_w):
                output[i, j] = np.sum(padded[i:i + k_h, j:j + k_w] * kernel)
        return output

    @staticmethod
    def reflect_pad(matrix, pad_h, pad_w):
        return np.pad(matrix, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    @staticmethod
    def hamming_distance(c1, c2):
        num_bands, bits_per_band, best_hd = 8, len(c1) // 8, 1.0
        for shift in range(-40, 41, 2):
            diff = 0
            for b in range(num_bands):
                s, e = b * bits_per_band, (b + 1) * bits_per_band
                diff += np.sum(c1[s:e] != np.roll(c2[s:e], shift))
            best_hd = min(best_hd, diff / len(c1))
        return best_hd

    @staticmethod
    def code_to_image(code):
        width = len(code) // 8
        return code[:8 * width].reshape((8, width)).astype(np.uint8) * 255

    def _load_code_database(self):
        manifest = os.path.join("iris_codes", "manifest.csv")
        if not os.path.isfile(manifest): return {}
        db = {}
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if os.path.isfile(row["code_file"]):
                    db[(row["person_id"], row["side"])] = np.load(row["code_file"])
        return db


if __name__ == "__main__":
    root = tk.Tk();
    app = BiometriaApp(root);
    root.mainloop()