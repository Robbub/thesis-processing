import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile
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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)