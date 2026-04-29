
import os
import sys
import csv
import numpy as np
from app import BiometriaApp as ba
from utils import process_eye_projections_pro



def first_image_in(folder):
    if not os.path.isdir(folder):
        return None
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
    )
    return os.path.join(folder, files[0]) if files else None


def generate_codes(
    train_dir="train-subset",
    output_dir="iris_codes",
    freq=0.2892, # wybrane eksperymentalnie
    convolve_maker="scipy",
):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    person_dirs = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )

    rows = []

    for person_id in person_dirs:
        person_path = os.path.join(train_dir, person_id)

        for side in ("left", "right"):
            side_folder = os.path.join(person_path, side)
            img_path = first_image_in(side_folder)


            print(f"Osoba {person_id}, {side} -> {os.path.basename(img_path)}")

            result = process_eye_projections_pro(img_path)
            if result is None:
                raise ValueError(f"problem z {person_id}, {side}")

            _orig, _m_p, _m_i, _det, flat = result

            code = ba.iris_code(flat, freq=freq, convolve_maker=convolve_maker)

            code_filename = f"person_{int(person_id):02d}_{side}.npy"
            code_path = os.path.join(output_dir, code_filename)
            np.save(code_path, code)

            rows.append({
                "person_id": person_id,
                "side": side,
                "source_file": os.path.abspath(img_path),
                "code_file": os.path.abspath(code_path),
                "code_length": len(code),
            })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["person_id", "side", "source_file", "code_file", "code_length"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generuj kody tęczówek dla train-subset.")
    parser.add_argument("--train_dir",      default="train-subset", help="Ścieżka do train-subset/")
    parser.add_argument("--output_dir",     default="iris_codes",   help="Folder wyjściowy na kody")
    parser.add_argument("--freq",           type=float, default=0.2892, help="Częstotliwość Gabora ")
    parser.add_argument("--convolve_maker", default="scipy",         help="'scipy' lub 'custom'")
    args = parser.parse_args()

    generate_codes(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        freq=args.freq,
        convolve_maker=args.convolve_maker,
    )