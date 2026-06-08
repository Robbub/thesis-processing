import os
import json
import shutil
import cv2
import numpy as np
from scipy import stats
from fastapi import UploadFile

STORAGE_MODE = os.environ.get("STORAGE_MODE", "LOCAL")
LOCAL_STORAGE_DIR = "storage"
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
if RENDER_URL:
    HOST_URL = f"{RENDER_URL.rstrip('/')}/static"
else:
    HOST_URL = "http://localhost:8000/static"


PIXEL_TO_MM = 0.1
GAP_THRESHOLD_PIXELS = 5

class CrackUnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def calculate_min_edge_distance(contour_a, contour_b):
    pts_a = contour_a.reshape(-1, 2)
    pts_b = contour_b.reshape(-1, 2)
    dist_matrix = np.linalg.norm(pts_a[:, np.newaxis] - pts_b, axis=2)
    return np.min(dist_matrix)

class InspectionRepository:

    @staticmethod
    def process_and_analyze_crack(orig_path, mask_path):
        img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"bounding_boxes": [], "contours": []}
        
        smoothed_full = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        whitehot_crack_full = cv2.adaptiveThreshold(
            smoothed_full, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 14
        )

        kernel_full = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        cleaned_full = cv2.morphologyEx(whitehot_crack_full, cv2.MORPH_OPEN, kernel_full)

        num_labels_f, labels_f, stats_f, _ = cv2.connectedComponentsWithStats(
            cleaned_full, connectivity=8, ltype=cv2.CV_32S
        )
        final_full_mask = np.zeros_like(cleaned_full)
        for i in range(1, num_labels_f):
            if stats_f[i, cv2.CC_STAT_AREA] >= 150:
                final_full_mask[labels_f == i] = 255

        h_full, w_full = whitehot_crack_full.shape
        web_mask = np.zeros((h_full, w_full, 4), dtype=np.uint8)
        web_mask[final_full_mask == 255] = [0, 0, 255, 255]
        web_mask[final_full_mask == 0] = [0, 0, 0, 0]

        cv2.imwrite(mask_path, web_mask)

        target_w, target_h = 416, 416
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        smoothed_small = cv2.bilateralFilter(resized_img, d=7, sigmaColor=50, sigmaSpace=50)
        whitehot_crack = cv2.adaptiveThreshold(
            smoothed_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 14
        )

        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        cleaned_mask = cv2.morphologyEx(whitehot_crack, cv2.MORPH_OPEN, kernel_small)

        num_labels, labels, stats_map, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8, ltype=cv2.CV_32S
        )
        final_cleaned_mask = np.zeros_like(cleaned_mask)
        for i in range(1, num_labels):
            if stats_map[i, cv2.CC_STAT_AREA] >= 75:
                final_cleaned_mask[labels == i] = 255

        contours, _ = cv2.findContours(final_cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        crack_records = []

        for count in contours:
            if cv2.contourArea(count) < 5:
                continue
        
            x, y, w, h = cv2.boundingRect(count)
            region_of_interest = np.zeros((h + 10, w + 10), dtype=np.uint8)
            shifted_count = count - [x -5, y - 5]
            cv2.drawContours(region_of_interest, [shifted_count], -1, 255, -1)
            dist_map = cv2.distanceTransform(region_of_interest, cv2.DIST_L2, 5)
            skeleton = cv2.ximgproc.thinning(region_of_interest, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            pixel_length = np.sum(skeleton == 255)
            length_mm = pixel_length * PIXEL_TO_MM

            if pixel_length == 0:
                continue

            widths_mm = dist_map[skeleton == 255] * 2.0 * PIXEL_TO_MM
            max_w = np.max(widths_mm)
            mean_w = np.mean(widths_mm)
            mode_w = float(stats.mode(np.round(widths_mm, 2), keepdims=True).mode[0])
            rotated_box = cv2.minAreaRect(count)
            orientation_angle = float(rotated_box[2])

            crack_records.append({
                "contour": count,
                "length_mm": length_mm,
                "max_width_mm": max_w,
                "mean_width_mm": mean_w,
                "mode_width_mm": mode_w,
                "orientation_deg": orientation_angle,
                "widths_raw": widths_mm
            })

        num_fragments = len(crack_records)
        if num_fragments == 0:
            return {"bounding_boxes": [], "contours": []}
        
        uf = CrackUnionFind(num_fragments)
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                gap = calculate_min_edge_distance(crack_records[i]["contour"], crack_records[j]["contour"])
                if gap <= GAP_THRESHOLD_PIXELS:
                    uf.union(i, j)

        clusters = {}
        for i in range(num_fragments):
            root = uf.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        final_boxes = []
        final_contours = []

        for chain_id, indices in enumerate(clusters.values()):
            sub_cracks = [crack_records[idx] for idx in indices]
            total_length = sum(c["length_mm"] for c in sub_cracks)
            max_width = max(c["max_width_mm"] for c in sub_cracks)
            mean_width = np.mean(np.concatenate([c["widths_raw"] for c in sub_cracks]))
            angles = np.array([c["orientation_deg"] for c in sub_cracks])
            lengths = np.array([c["length_mm"] for c in sub_cracks])

            if total_length > 0:
                percentage_weights = lengths / total_length
                weighted_average_angle = np.sum(angles * percentage_weights)

                within_tolerance = np.abs(angles - weighted_average_angle) <= 10.0

                if np.sum(within_tolerance) > (len(sub_cracks) / 2.0):
                    final_orientation = f"{weighted_average_angle:.1f} degrees"
                else:
                    final_orientation = "Curve"

            else:
                final_orientation = "Unknown"

            all_contour_points = np.concatenate([c["contour"] for c in sub_cracks], axis=0)

            bx, by, bw, bh = cv2.boundingRect(all_contour_points)
            pct_x = float((bx / target_w) * 100.0)
            pct_y = float((by / target_h) * 100.0)
            pct_w = float((bw / target_w) * 100.0)
            pct_h = float((bh / target_h) * 100.0)

            final_boxes.append({
                "id": int(chain_id + 1),
                "x": pct_x,
                "y": pct_y,
                "width": pct_w,
                "height": pct_h,
                "avgWidth": f"{mean_width:.2f} mm",
                "maxWidth": f"{max_width:.2f} mm",
                "crackLength": f"{total_length:.2f} mm",
                "orientation": str(final_orientation)
            })

            for c_idx, c in enumerate(sub_cracks):
                path_str = ""
                for idx, pt in enumerate(c["contour"]):
                    raw_x = float(pt[0][0])
                    raw_y = float(pt[0][1])
                    px = (raw_x / target_w) * 100.0
                    py = (raw_y / target_h) * 100.0
                    cmd = "M" if idx == 0 else "L"
                    path_str += f"{cmd} {px:.1f} {py:.1f} "

                final_contours.append({
                    "id": f"cont_{chain_id}_{c_idx}",
                    "path": path_str.strip()
                })

        return {
            "bounding_boxes": final_boxes,
            "contours": final_contours
        }


    
    @staticmethod
    def save_new_inspection(file: UploadFile, file_id: str) -> dict:
        if STORAGE_MODE == "LOCAL":
            folder_path = os.path.join(LOCAL_STORAGE_DIR, file_id)
            os.makedirs(folder_path, exist_ok=True)

            cleaned_base_name, incoming_extension = os.path.splitext(file.filename)
            orig_filename = f"original_{cleaned_base_name}{incoming_extension}"
            mask_filename = f"mask_{cleaned_base_name}.png"
            orig_path = os.path.join(folder_path, orig_filename)
            mask_path = os.path.join(folder_path, mask_filename)
            
            with open(orig_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            crack_data = InspectionRepository.process_and_analyze_crack(orig_path, mask_path)

            meta_payload = {
                "id": file_id,
                "name": file.filename,
                "original_url": f"{HOST_URL}/{file_id}/{orig_filename}",
                "mask_url": f"{HOST_URL}/{file_id}/{mask_filename}",
                "crack_data": crack_data
            }

            with open(os.path.join(folder_path, "crack_meta.json"), "w") as json_file:
                json.dump(meta_payload, json_file)

            return meta_payload
        
        elif STORAGE_MODE == "CLOUD":
            #TODO: fetch from DAVE + KC
            pass

    @staticmethod
    def get_all_inspections() -> list:

        if STORAGE_MODE == "LOCAL":
            inspections_list = []
            if not os.path.exists(LOCAL_STORAGE_DIR):
                return []
            
            for folder_name in os.listdir(LOCAL_STORAGE_DIR):
                folder_path = os.path.join(LOCAL_STORAGE_DIR, folder_name)
                if os.path.isdir(folder_path):
                    meta_file = os.path.join(folder_path, "crack_meta.json")

                    if os.path.exists(meta_file):
                        with open(meta_file, "r") as f:
                            meta_data = json.load(f)
                            inspections_list.append(meta_data)
            return inspections_list
    
        elif STORAGE_MODE == "CLOUD":
            #TODO: fetch
            pass