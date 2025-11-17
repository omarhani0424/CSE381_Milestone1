"""
Milestone 1 pipeline for jigsaw puzzle piece segmentation (Classical CV, no ML).
Designed to run on Google Colab or locally. 
Outputs:
 - cropped piece images (./output/crops)
 - piece masks (./output/masks)
 - descriptor JSON file (./output/descriptors.json)

Usage in Colab (quick):
1. Upload an image (or mount Drive):
   from google.colab import files
   files.upload()  # choose image(s)
2. Run this script in Colab:
   !python3 /content/milestone1_pipeline.py --input /content/your_image.jpg

Or import the functions from this file in a notebook and call pipeline(image).
"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# ------------------------ Utilities ------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

# ------------------------ Core pipeline functions ------------------------
def denoise_image(img: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Denoise while preserving edges. methods: 'gaussian', 'median', 'bilateral' (default)
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (5,5), 0)
    elif method == 'median':
        return cv2.medianBlur(img, 5)
    else:
        # bilateral preserves edges
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def enhance_contrast(img: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Enhance contrast using CLAHE on the luminance channel.
    """
    if len(img.shape) == 2:
        # grayscale
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(img)
        else:
            return img
    # convert to LAB and apply CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
    else:
        cl = l
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def edge_enhance(img: np.ndarray) -> np.ndarray:
    """
    Simple unsharp masking to slightly sharpen edges.
    """
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened

def background_remove(img: np.ndarray, debug: bool=False) -> np.ndarray:
    """
    Convert to grayscale, apply adaptive thresholding or Otsu depending on image.
    Returns a binary mask where foreground (pieces) = 255.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Try Otsu threshold first
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # otsu can sometimes invert foreground/background; check which side likely contains objects
    # Heuristic: count non-zero pixels in center crop; if center has background, invert
    h, w = otsu_mask.shape
    cx, cy = w//2, h//2
    center_patch = otsu_mask[cy-10:cy+10, cx-10:cx+10]
    # if center patch is mostly white, we assume background is white, invert to make pieces white
    if np.mean(center_patch) > 127:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small noise components using connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    # Filter by area: keep components larger than a fraction of image
    min_area = max(500, int(0.0005 * h * w))  # heuristic
    mask = np.zeros_like(clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255

    if debug:
        return mask, gray, otsu_mask, clean
    return mask

def find_piece_contours(binary_mask: np.ndarray, min_area: int = None) -> List[np.ndarray]:
    """
    Find external contours in the binary mask and filter by area.
    Returns list of contours (each is Nx1x2 array).
    """
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if min_area is None:
        h, w = binary_mask.shape
        min_area = max(1000, int(0.001 * h * w))
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    # sort by area descending (largest first)
    filtered = sorted(filtered, key=cv2.contourArea, reverse=True)
    return filtered

def crop_piece_and_mask(img: np.ndarray, contour: np.ndarray, padding: int = 8) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """
    Given full image and contour, return cropped image, cropped mask, and bbox (x,y,w,h)
    """
    x,y,w,h = cv2.boundingRect(contour)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img.shape[1], x + w + padding)
    y1 = min(img.shape[0], y + h + padding)
    crop = img[y0:y1, x0:x1].copy()
    # translate contour to crop coords
    contour_c = contour.copy()
    contour_c[:,0,0] -= x0
    contour_c[:,0,1] -= y0
    mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    cv2.drawContours(mask, [contour_c], -1, 255, thickness=-1)
    return crop, mask, (x0, y0, x1-x0, y1-y0)

def extract_edge_points_from_contour(contour: np.ndarray, epsilon_factor: float = 0.002) -> List[Tuple[int,int]]:
    """
    Optionally approximate contour to get fewer edge points. Returns list of (x,y).
    epsilon_factor controls approximation; smaller -> closer fit.
    """
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = [tuple(int(v) for v in p[0]) for p in approx]
    return pts

def store_descriptors(descriptors: Dict, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    save_json(descriptors, out_path)

# ------------------------ High-level pipeline ------------------------
def pipeline_process_image(input_path: str, output_dir: str, debug: bool = False) -> Dict:
    """
    Full pipeline: read image -> denoise -> enhance -> background remove -> find contours -> crop + save -> save descriptors.
    Returns descriptors dictionary.
    """
    ensure_dir(output_dir)
    crops_dir = os.path.join(output_dir, 'crops')
    masks_dir = os.path.join(output_dir, 'masks')
    ensure_dir(crops_dir)
    ensure_dir(masks_dir)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    # Resize large images for faster processing (keeping aspect ratio)
    max_dim = 1600
    h0, w0 = img.shape[:2]
    if max(h0, w0) > max_dim:
        scale = max_dim / float(max(h0, w0))
        img = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

    den = denoise_image(img, method='bilateral')
    enhanced = enhance_contrast(den, use_clahe=True)
    sharp = edge_enhance(enhanced)

    mask = background_remove(sharp)
    contours = find_piece_contours(mask)

    descriptors = {}
    for idx, c in enumerate(contours):
        piece_id = f"piece_{idx+1:03d}"
        crop, crop_mask, bbox = crop_piece_and_mask(img, c, padding=10)
        # Save images
        crop_path = os.path.join(crops_dir, f"{piece_id}.png")
        mask_path = os.path.join(masks_dir, f"{piece_id}_mask.png")
        cv2.imwrite(crop_path, crop)
        cv2.imwrite(mask_path, crop_mask)
        # Contour coordinates in original image coordinates
        contour_coords = [[int(pt[0][0]), int(pt[0][1])] for pt in c]
        edge_points = extract_edge_points_from_contour(c)
        descriptors[piece_id] = {
            "id": piece_id,
            "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])},
            "area": float(cv2.contourArea(c)),
            "contour": contour_coords,
            "edge_points": edge_points,
            "crop_path": os.path.abspath(crop_path),
            "mask_path": os.path.abspath(mask_path)
        }

    # Save descriptors JSON
    desc_path = os.path.join(output_dir, 'descriptors.json')
    store_descriptors(descriptors, desc_path)
    if debug:
        return {"descriptors": descriptors, "mask": mask, "enhanced": enhanced}
    return {"descriptors": descriptors}

# ------------------------ Command-line interface ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Milestone1 puzzle segmentation pipeline")
    p.add_argument("--input", "-i", required=True, help="Input image path")
    p.add_argument("--output", "-o", default="./output", help="Output directory")
    p.add_argument("--debug", action="store_true", help="Save extra debug images")
    return p.parse_args()

def main():
    args = parse_args()
    # create output dir
    ensure_dir(args.output)
    result = pipeline_process_image(args.input, args.output, debug=args.debug)
    print("Saved descriptors:", os.path.join(args.output, 'descriptors.json'))
    print("Crops:", os.path.join(args.output, 'crops'))
    print("Masks:", os.path.join(args.output, 'masks'))

if __name__ == "__main__":
    main()
