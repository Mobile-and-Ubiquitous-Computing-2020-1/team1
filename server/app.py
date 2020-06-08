"""FedEx Server."""

import asyncio
import os
import sys

import aiofiles
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Body
from fastapi.responses import FileResponse, StreamingResponse
from starlette.responses import PlainTextResponse
from pydantic.dataclasses import dataclass


import const as C
from trainer import Trainer, get_trainer, register


@dataclass
class Accuracy:
  trial_id: int
  acc: float


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
    """Respond pong."""
    return "pong"


  @app.post(C.PUSH_URL)
  async def _push_intermediate_features(
    request: Request,
    *,
    content_type: str = Header(None),
    content_disposition: str = Header(None),
    trainer: Trainer = Depends(get_trainer)
  ):
    """Push new features to server."""
    if content_type != "application/octet-stream":
      raise HTTPException(status_code=400, detail="wrong header, should be 'application/octet-stream'")

    await trainer.add_intermediate_feature(request)
    return dict(success=True)


  @app.get(C.PULL_URL)
  async def _pull_model(
    trainer: Trainer = Depends(get_trainer)
  ):
    """Pull models from server."""
    filename = trainer.get_best_model_file()
    fullname = os.path.join(C.MODEL_PATH, filename)
    return FileResponse(fullname, media_type="application/octet-stream")


  @app.get(C.INFO_URL)
  async def _info_model(
    trainer: Trainer = Depends(get_trainer)
  ):
    """Get information of best model."""
    model_info = trainer.get_best_model_info()
    return model_info

  @app.post(C.UPDATE_URL)
  async def _update_model_perf(
    acc: Accuracy = Body(...),
    trainer: Trainer = Depends(get_trainer)
  ):
    """Update a running model performance."""
    print('update trial')
    trainer.update_model(acc)

  @app.get(C.LIST_URL)
  async def _list_models(
    trainer: Trainer = Depends(get_trainer)
  ):
    return trainer.trained_models

  return app



app = app_generator()
