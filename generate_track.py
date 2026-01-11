import cv2
import numpy as np
import csv
import yaml
import os
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
# The name of your map (without extension)
MAP_NAME = "FTMHalle_ws25" 

# Base path to your package (adjust if your path is different)
BASE_PATH = "/root/sim_ws/src/f1tenth_gym_ros"
# ---------------------

def main():
    # Construct full paths
    pgm_path = os.path.join(BASE_PATH, "maps", f"{MAP_NAME}.pgm")
    yaml_path = os.path.join(BASE_PATH, "maps", f"{MAP_NAME}.yaml")
    out_path = os.path.join(BASE_PATH, "racelines", f"{MAP_NAME}.csv")

    print(f"Reading map from: {pgm_path}")

    # 1. Read Map Metadata (YAML) to get resolution and origin
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    resolution = data['resolution']
    origin = data['origin'] # [x, y, z]

    # 2. Read Map Image (PGM)
    # Load as grayscale (0)
    img = cv2.imread(pgm_path, 0)
    if img is None:
        print(f"Error: PGM file not found at {pgm_path}")
        return

    # 3. Process Image
    # Flip vertical because OpenCV (0,0 is Top-Left) but ROS map (0,0 is Bottom-Left)
    img_flipped = cv2.flip(img, 0)

    # Threshold: Convert to binary. White (Free) = 1, Black (Wall) = 0
    # We use 200 as the threshold (light grey and white become free space)
    _, binary_img = cv2.threshold(img_flipped, 200, 1, cv2.THRESH_BINARY)

    # Skeletonize: Shrink the white path down to a 1-pixel wide centerline
    skeleton = skeletonize(binary_img)

    # 4. Extract Waypoints
    # Get the coordinates of the white pixels (the centerline)
    # numpy returns (row, col) which corresponds to (y, x)
    y_pixels, x_pixels = np.where(skeleton > 0)

    waypoints = []
    for x_px, y_px in zip(x_pixels, y_pixels):
        # Convert Pixel -> Meters
        # formula: world_pos = origin + (pixel_coord * resolution)
        wx = origin[0] + (x_px * resolution)
        wy = origin[1] + (y_px * resolution)
        waypoints.append([wx, wy])

    # 5. Sort the Points (Nearest Neighbor)
    # Skeletonize gives us pixels in random order. We must chain them into a loop.
    if len(waypoints) == 0:
        print("Error: No path found! Is the map image mostly black?")
        return

    print(f"Found {len(waypoints)} raw points. Sorting...")

    sorted_pts = [waypoints[0]]
    waypoints.pop(0)

    while len(waypoints) > 0:
        current_pt = sorted_pts[-1]
        
        # Calculate distances to all remaining points
        dists = np.sum((np.array(waypoints) - np.array(current_pt))**2, axis=1)
        nearest_idx = np.argmin(dists)
        
        # Add nearest point to the sorted list
        sorted_pts.append(waypoints[nearest_idx])
        waypoints.pop(nearest_idx)

    # 6. Save to CSV
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        # Header required by some parsers
        writer.writerow(['x', 'y', 'yaw', 'speed'])

        for i in range(len(sorted_pts)):
            curr = sorted_pts[i]
            
            # Calculate Yaw (Direction to next point)
            # Use modulo % to wrap around to the first point at the end
            next_pt = sorted_pts[(i + 5) % len(sorted_pts)] # Look 5 points ahead for smoother yaw
            
            dx = next_pt[0] - curr[0]
            dy = next_pt[1] - curr[1]
            yaw = np.arctan2(dy, dx)
            
            # Set a conservative speed (e.g., 2.0 m/s)
            speed = 2.0
            
            writer.writerow([curr[0], curr[1], yaw, speed])

    print("------------------------------------------------")
    print(f"SUCCESS! Raceline generated with {len(sorted_pts)} points.")
    print(f"File saved to: {out_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()
