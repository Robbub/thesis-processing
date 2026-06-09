import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from storage import InspectionRepository

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="storage"), name="static")

@app.get("/api/inspections")
async def list_inspections():
    return InspectionRepository.get_all_inspections()

@app.post("/api/upload")
async def upload_inspection(file: Optional[UploadFile] = File(None)):
    if file is None:
        return HTTPException(status_code=400, detail="No local file stream provided for processing.")
    
    unique_id = f"insp_{int(time.time())}"
    return InspectionRepository.save_new_inspection(file, unique_id)
    
@app.post("/api/analyze-session")
async def analyze_session(payload: dict = Body(...)):
    session_id = payload.get("sessionId")
    originals = payload.get("originals", [])

    if not session_id or not originals:
        raise HTTPException(status_code=400, detail="Missing sessionId or original assets list.")
    
    analyzed_originals = []

    for orig in originals:
        orig_id = orig.get("id")
        orig_url = orig.get("url") or orig.get("storageUrl") or orig.get("original_url")

        resized_variants = orig.get("resized_variants", []) or []
        if isinstance(resized_variants, list) and resized_variants:
            variant_target = resized_variants[0]
            resized_url = variant_target.get("url") or variant_target.get("storageUrl") or variant_target.get("original_url")
        else:
            resized_url = orig_url

        if not resized_url:
            print(f"Skipping original {orig_id} because no source URL is available.")
            orig["is_processed"] = False
            analyzed_originals.append(orig)
            continue

        try:
            result = InspectionRepository.process_cloud_session_images(
                original_id=orig_id,
                original_url=orig_url,
                resized_url=resized_url
            )

            orig["mask_url"] = result.get("mask_url")
            orig["crack_data"] = result.get("crack_data", {"bounding_boxes": [], "contours": []})
            orig["is_processed"] = True
        except Exception as e:
            print(f"Failed to process asset {orig_id}: {str(e)}")
            orig["is_processed"] = False

        analyzed_originals.append(orig)

    processed_session = {
        "sessionId": session_id,
        "originals": analyzed_originals,
        "is_processed_session": True
    }

    if hasattr(InspectionRepository, "save_session_metadata"):
        InspectionRepository.save_session_metadata(processed_session)

    return processed_session

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)