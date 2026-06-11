import base64
import os
import json
import shutil
import requests
import cv2
import numpy as np
from scipy import stats
from fastapi import UploadFile
import firebase_admin
from firebase_admin import credentials, firestore

STORAGE_MODE = os.environ.get("STORAGE_MODE", "LOCAL")
LOCAL_STORAGE_DIR = "storage"
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
if RENDER_URL:
    HOST_URL = f"{RENDER_URL.rstrip('/')}/static"
else:
    HOST_URL = "http://localhost:8000/static"

MASK_API_URL = os.environ.get(
    "DAMAGE_MASK_API_URL",
    "https://damage-mask-service.onrender.com/process-all"
)

PIXEL_TO_MM = 0.1
GAP_THRESHOLD_PIXELS = 5

if STORAGE_MODE == "CLOUD":
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)

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
    dist_matrix = np.linalg.norm(pts_a[:2, np.newaxis] - pts_b, axis=2)
    return np.min(dist_matrix)

class InspectionRepository:

    @staticmethod
    def process_and_analyze_crack(mask_matrix=None):

        if mask_matrix is None:
            return {"bounding_boxes": [], "contours": []}

        target_w, target_h = 416, 416
        if mask_matrix.shape[:2] != (target_h, target_w):
            processed_mask = cv2.resize(mask_matrix, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            processed_mask = mask_matrix

        _, cleaned_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)

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
            mode_result = stats.mode(np.round(widths_mm, 2), keepdims=True).mode
            mode_w = float(mode_result.item() if hasattr(mode_result, "item") else mode_result)
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

        running_total_length = 0.0
        running_all_widths = []

        for chain_id, indices in enumerate(clusters.values()):
            sub_cracks = [crack_records[idx] for idx in indices]
            total_length = sum(c["length_mm"] for c in sub_cracks)
            max_width = max(c["max_width_mm"] for c in sub_cracks)
            concatenated_widths = np.concatenate([c["widths_raw"] for c in sub_cracks])
            mean_width = np.mean(np.concatenate([c["widths_raw"] for c in sub_cracks]))
            running_total_length += total_length
            running_all_widths.append(concatenated_widths)
            angles = np.abs(np.array([c["orientation_deg"] for c in sub_cracks]))
            lengths = np.array([c["length_mm"] for c in sub_cracks])

            if total_length > 0:
                percentage_weights = lengths / total_length
                weighted_average_angle = np.sum(angles * percentage_weights)
                within_tolerance = np.abs(angles - weighted_average_angle) <= 10.0

                if np.sum(within_tolerance) > (len(sub_cracks) / 2.0):
                    final_orientation = f"{np.abs(weighted_average_angle):.1f} degrees"
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

            current_box_id = int(chain_id + 1)

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
                    "parent_box_id": current_box_id,
                    "path": path_str.strip()
                })

        overall_max_width = max([box.get("maxWidth").split()[0] for box in final_boxes]) if final_boxes else "0.00"
        macro_stats = {
            "total_length_mm": round(running_total_length, 2),
            "average_width_mm": round(float(np.mean(np.concatenate(running_all_widths))), 2) if running_all_widths else 0.0,
            "max_width_mm": round(float(overall_max_width), 2)
        }

        return {
            "macro_stats": macro_stats,
            "bounding_boxes": final_boxes,
            "contours": final_contours
        }
    
    @staticmethod
    def process_cloud_session_images(original_id: str, original_url: str, resized_url: str) -> dict:
        external_api_url = MASK_API_URL
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        payload = {
            "original_id": str(original_id),
            "original_url": str(original_url),
            "resized_url": str(resized_url)
        }

        print(f"Triggering background batch process via: {external_api_url}")

        try:
            api_response = requests.post(external_api_url, json=payload, headers=headers, timeout=15)
            print(f"AI Batch Service Initial Response Status: {api_response.status_code}")
        except Exception as e:
            print(f"AI Service Wakeup notice sent. (Handled error/timeout): {e}")

        return {
            "mask_url": None,
            "crack_data": {"bounding_boxes": [], "contours": []}
        }


    @staticmethod
    def save_new_inspection(file: UploadFile, file_id: str) -> dict:
        if STORAGE_MODE == "LOCAL":
            folder_path = os.path.join(LOCAL_STORAGE_DIR, file_id)
            os.makedirs(folder_path, exist_ok=True)
            os.chmod(folder_path, 0o755)

            cleaned_base_name, incoming_extension = os.path.splitext(file.filename)
            orig_filename = f"original_{cleaned_base_name}{incoming_extension}"
            orig_path = os.path.join(folder_path, orig_filename)
            
            with open(orig_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            session_payload = {
                "sessionId": file_id,
                "originals": [
                    {
                        "id": file_id,
                        "name": file.filename,
                        "url": f"{HOST_URL}/{file_id}/{orig_filename}",
                        "mask_url": None,
                        "crack_data": {"bounding_boxes": [], "contours": []},
                        "resized_variants": []
                    }
                ],
                "is_processed_session": False
            }

            with open(os.path.join(folder_path, "session_meta.json"), "w") as json_file:
                json.dump(session_payload, json_file)

            return session_payload
        
        elif STORAGE_MODE == "CLOUD":
            raise NotImplementedError("CLOUD storage uploads are not implemented. Set STORAGE_MODE=LOCAL or add cloud upload support.")

    @staticmethod
    def save_cloud_inspection(image_url: str, filename: str, file_id: str) -> dict:
        return {
            "sessionId": file_id,
            "originals": [
                {
                    "id": file_id,
                    "name": filename,
                    "url": image_url,
                    "mask_url": None,
                    "crack_data": {"bounding_boxes": [], "contours": []},
                    "resized_variants": []
                }
            ],
            "is_processed_session": False
        }

    @staticmethod
    def save_session_metadata(session_payload: dict) -> dict:
        if STORAGE_MODE != "LOCAL":
            return session_payload

        folder_path = os.path.join(LOCAL_STORAGE_DIR, session_payload.get("sessionId", ""))
        os.makedirs(folder_path, exist_ok=True)
        meta_file = os.path.join(folder_path, "session_meta.json")

        with open(meta_file, "w") as f:
            json.dump(session_payload, f)

        return session_payload

    @staticmethod
    def get_all_inspections() -> list:

        if STORAGE_MODE == "LOCAL":
            inspections_list = []
            if not os.path.exists(LOCAL_STORAGE_DIR):
                return []
            
            for folder_name in os.listdir(LOCAL_STORAGE_DIR):
                folder_path = os.path.join(LOCAL_STORAGE_DIR, folder_name)
                if os.path.isdir(folder_path):
                    meta_file = os.path.join(folder_path, "session_meta.json")
                    if not os.path.exists(meta_file):
                        meta_file = os.path.join(folder_path, "crack_meta.json")

                    if os.path.exists(meta_file):
                        with open(meta_file, "r") as f:
                            meta_data = json.load(f)
                            inspections_list.append(meta_data)
            return inspections_list
    
        elif STORAGE_MODE == "CLOUD":
            db = firestore.client()
            images_ref = db.collection("images")
            docs = images_ref.stream()

            sessions_map = {}
            originals_map = {}
            resized_list = []

            for doc in docs:
                data = doc.to_dict()
                doc_id = data.get("original_id") or doc.id
                data["id"] = doc_id

                s_id = data.get("sessionId")
                img_type = data.get("type")

                if not s_id:
                    continue

                if s_id not in sessions_map:
                    sessions_map[s_id] = {
                        "sessionId": s_id,
                        "originals": []
                    }

                if img_type == "original":
                    data["resized_variants"] = []
                    data["url"] = data.get("storageUrl") or data.get("url")
                    
                    cloud_mask_url = data.get("maskS3Url") or data.get("mask_url")
                    data["mask_url"] = cloud_mask_url

                    has_empty_crack_data = ("crack_data" not in data or 
                                            not data["crack_data"].get("bounding_boxes") or 
                                            len(data["crack_data"]["bounding_boxes"]) == 0)

                    if cloud_mask_url and has_empty_crack_data:
                        print(f"New cloud mask discovered for asset {doc_id}. Compiling OpenCV measurements...")
                        try:

                            mask_resp = requests.get(cloud_mask_url, timeout=15)
                            mask_resp.raise_for_status()
                            mask_bytes = mask_resp.content
                            
                            mask_matrix = cv2.imdecode(np.asarray(bytearray(mask_bytes), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                            
                            if mask_matrix is not None and mask_matrix.size > 0:
                                calculated_crack_data = InspectionRepository.process_and_analyze_crack(mask_matrix=mask_matrix)
                                
                                data["crack_data"] = calculated_crack_data
                                
                                images_ref.document(doc.id).update({
                                    "crack_data": calculated_crack_data
                                })
                                print(f"Successfully cached crack assessment parameters for doc {doc.id} in cloud.")
                        except Exception as e:
                            print(f"Failed to process and analyze newly found mask: {e}")
                    
                    if "crack_data" not in data:
                        data["crack_data"] = {"bounding_boxes": [], "contours": []}

                    if doc_id:
                        originals_map[doc_id] = data
                    sessions_map[s_id]["originals"].append(data)
                    
                elif img_type == "resized":
                    data["url"] = data.get("storageUrl") or data.get("url")
                    data["mask_url"] = data.get("maskS3Url") or None
                    resized_list.append(data)

            for r_img in resized_list:
                parent_id = r_img.get("original_id")
                if parent_id in originals_map:
                    originals_map[parent_id]["resized_variants"].append(r_img)
            
            return list(sessions_map.values())
