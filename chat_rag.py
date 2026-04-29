from fastapi import FastAPI ,UploadFile, File,HTTPException
import os
import shutil
 
app = FastAPI()

Upload_Dir = "uplaods"
if not os.path.exists(Upload_Dir):
    os.makedirs(Upload_Dir)
