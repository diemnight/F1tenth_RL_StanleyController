import cv2
import numpy as np
import yaml
import csv
import os
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

# ================= CONFIG =================
MAP_NAME = "Spielberg_map"
BASE_PATH = "/root/sim_ws/src/f1tenth_gym_ros"
YAW_LOOKAHEAD = 10
MIN_POINTS = 1000
# =========================================

def order_loop(points):
    """
    Robust loop ordering using nearest-neighbor walk
    Assumes ONE continuous skeleton loop
    """
    pts = points.copy()
    ordered = [pts[0]]
    used = np.zeros(len(pts), dtype=bool)
    used[0] = True

    tree = KDTree(pts)

    for _ in range(len(pts) - 1):
        last = ordered[-1]
        dists, idxs = tree.query(last, k=6)
        next_idx = None
        for i in idxs:
            if not used[i]:
                next_idx = i
                break
        if next_idx is None:
            break
        ordered.append(pts[next_idx])
        used[next_idx] = True

    return np.array(ordered)

def main():
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    img_path = os.path.join(BASE_PATH, "maps", f"{MAP_NAME}.png")
    yaml_path = os.path.join(BASE_PATH, "maps", f"{MAP_NAME}.yaml")
    out_path = os.path.join(BASE_PATH, "racelines", f"{MAP_NAME}.csv")

    # --------------------------------------------------
    # Load map metadata
    # --------------------------------------------------
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    resolution = meta["resolution"]
    origin = meta["origin"]

    # --------------------------------------------------
    # Load binary map (TRACK ONLY)
    # --------------------------------------------------
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to load map image")

    img = cv2.flip(img, 0)
    track = (img > 200).astype(np.uint8)

    # --------------------------------------------------
    # Skeletonize (CENTERLINE)
    # --------------------------------------------------
    skel = skeletonize(track).astype(np.uint8)
    ys, xs = np.where(skel > 0)

    print(f"Centerline pixels: {len(xs)}")
    if len(xs) < MIN_POINTS:
        raise RuntimeError("Skeleton too small — map not filled correctly")

    # --------------------------------------------------
    # Convert to world coordinates
    # --------------------------------------------------
    pts = []
    for x, y in zip(xs, ys):
        wx = origin[0] + x * resolution
        wy = origin[1] + y * resolution
        pts.append([wx, wy])

    pts = np.array(pts)

    # --------------------------------------------------
    # Order into single loop
    # --------------------------------------------------
    ordered = order_loop(pts)

    # --------------------------------------------------
    # Compute yaw
    # --------------------------------------------------
    yaws = []
    n = len(ordered)
    for i in range(n):
        p = ordered[i]
        q = ordered[(i + YAW_LOOKAHEAD) % n]
        yaw = np.arctan2(q[1] - p[1], q[0] - p[0])
        yaws.append(yaw)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "yaw"])
        for (x, y), yaw in zip(ordered, yaws):
            writer.writerow([x, y, yaw])

    print("===================================")
    print("CENTERLINE GENERATED SUCCESSFULLY")
    print(f"Points: {len(ordered)}")
    print(f"Saved to: {out_path}")
    print("===================================")

if __name__ == "__main__":
    main()
