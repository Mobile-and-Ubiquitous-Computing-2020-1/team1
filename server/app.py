"""FedEx Server."""

import asyncio
import os

import aiofiles
import sys

from starlette.responses import PlainTextResponse
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse


import const as C


def app_generator() -> FastAPI:
  app = FastAPI(
    title="FedEx server",
    description="FedEx push/pull server",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
  )

  @app.get(C.PING_URL, response_class=PlainTextResponse)
  def _pong() -> str:
    return "pong"


  @app.post(C.PUSH_URL)
  async def _push_intermediate_features(request: Request, *, content_type: str = Header(None), content_disposition: str = Header(None)):
    if content_type != "application/octet-stream":
      raise HTTPException(status_code=400, detail="wrong header, should be 'application/octet-stream'")

    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '../intermediate-features')
    filename = content_disposition.split("filename=")[1].replace('"', '')
    fullname = os.path.join(dirname, filename)

    os.makedirs(dirname, exist_ok=True)

    async with aiofiles.open(fullname, 'wb') as output:
      await output.write(await request.body())

    asyncio.create_task(_background_train())
    return dict(success=True)


  @app.get(C.PULL_URL)
  async def _pull_model():
    TFLITE_MODEL_DIR = 'tflite-models'
    filename = 'facenet_new.tflite'
    fullname = os.path.join(TFLITE_MODEL_DIR, filename)
    return FileResponse(fullname, media_type="application/octet-stream")

  return app

app = app_generator()

async def _background_train():
  command = "../bg.py"
  await asyncio.create_subprocess_exec(sys.executable, command)
