#!/usr/bin/env python3
import argparse
import cv2
import numpy as np


def shift_2d_replace(data, dx, dy, v=0):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = v
    elif dx > 0:
        shifted_data[:, 0:dx] = v

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = v
    elif dy > 0:
        shifted_data[0:dy, :] = v
    return shifted_data


def main():
    parser = argparse.ArgumentParser("Fractal flame to heightmap converter")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--gamma1", type=float, default=2.4)
    parser.add_argument("--gamma2", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=0.125)
    args = parser.parse_args()

    # Load & Preprocess
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")
    img = img / 255
    r, g, b, a = [img[:, :, i] for i in range(4)]

    # Gamma Correction
    a **= 1 / args.gamma1

    # Merge (naive r+g+b)
    v = (r * 19595 + g * 38470 + b * 7471) * a / 65536

    # Delta-processing
    n = int(1 // args.delta)
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            d = (dx**2 + dy**2) ** 0.5 * args.delta
            vd = shift_2d_replace(v - d, dx, dy)
            v = np.maximum(v, vd)

    # Second Gamma Correction
    v **= 1 / args.gamma2

    # Dump
    v = (v * 255 + 0.5).astype("uint8")
    cv2.imwrite(args.output, v)


if __name__ == "__main__":
    main()
