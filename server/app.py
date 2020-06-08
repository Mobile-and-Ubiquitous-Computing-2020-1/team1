"""FedEx Server."""

import asyncio
import os
import sys

import aiofiles
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.responses import PlainTextResponse
from pydantic.dataclasses import dataclass


import const as C
from trainer import Trainer, get_trainer, register


def app_generator() -> FastAPI:
  app = FastAPI(
    title="FedEx server",
    description="FedEx push/pull server",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
  )
  register(app)

  @app.get(C.PING_URL, response_class=PlainTextResponse)
  def _pong() -> str:
    return "pong"


  @app.post(C.PUSH_URL)
  async def _push_intermediate_features(
    request: Request,
    *,
    content_type: str = Header(None),
    content_disposition: str = Header(None),
    trainer: Trainer = Depends(get_trainer)
  ):
    if content_type != "application/octet-stream":
      raise HTTPException(status_code=400, detail="wrong header, should be 'application/octet-stream'")

    filename = content_disposition.split("filename=")[1].replace('"', '')
    fullname = os.path.join(C.FEATURE_PATH, filename)

    async with aiofiles.open(fullname, 'wb') as output:
      await output.write(await request.body())  # todo change to streaming

    trainer.add_intermediate_feature(filename)
    return dict(success=True)


  @app.get(C.PULL_URL)
  async def _pull_model(
    trainer: Trainer = Depends(get_trainer)
  ):
    filename = trainer.get_best_model_file()
    fullname = os.path.join(C.MODEL_PATH, filename)
    return FileResponse(fullname, media_type="application/octet-stream")


  @app.get(C.INFO_URL)
  async def _info_model(
    trainer: Trainer = Depends(get_trainer)
  ):
    model_info = trainer.get_best_model_info()
    return model_info

  @app.post(C.UPDATE_URL)
  async def _update_model_perf(
    trainer: Trainer = Depends(get_trainer)
  ):
    trainer.update_model()

  return app


@dataclass
class A:
  pass


app = app_generator()
