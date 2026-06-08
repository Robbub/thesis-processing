import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
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
async def upload_inspection(
    file: Optional[UploadFile] = File(None),
    request: Request = None
):
    unique_id = f"insp_{int(time.time())}"

    if file is not None:
        return InspectionRepository.save_new_inspection(file, unique_id)
    else:
        body = await request.json()
        image_url = body.get("image_url")
        filename = body.get("filename", "cloud_image.jpg")

        return InspectionRepository.save_cloud_inspenction(image_url, filename, unique_id)
    
@app.post("/api/analyze-session")
async def analyze_session(payload: dict = Body(...)):
    session_id = payload.get("sessionId")
    originals = payload.get("originals", [])

    if not session_id or not originals:
        raise HTTPException(status_code=400, detail="Missing sessionId or original assets list.")
    
    analyzed_originals = []

    for orig in originals:
        orig_id = orig.get("id")
        orig_url = orig.get("url") or orig.get("storageUrl")

        resized_variants = orig.get("resized_variants", [])
        if not resized_variants:
            continue

        resized_url = resized_variants[0].get("url") or resized_variants[0].get("storageUrl")

        try:
            result = InspectionRepository.process_cloud_session_images(
                original_id=orig_id,
                original_url=orig_url,
                resized_url=resized_url
            )

            orig["mask_url"] = result["mask_url"]
            orig["crack_data"] = result["crack_data"]
            orig["is_processesd"] = True

        except Exception as e:
            print(f"Failed to process asset {orig_id}: {str(e)}")
            orig["is_processed"] = False
            
        analyzed_originals.append(orig)

    return {
        "sessionId": session_id,
        "originals": analyzed_originals,
        "is_processed_session": True
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)