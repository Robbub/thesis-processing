import time
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
async def upload_inspection(file: UploadFile = File(...)):
    unique_id = f"insp_{int(time.time())}"
    return InspectionRepository.save_new_inspection(file, unique_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)