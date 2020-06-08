"""Trainer component.

Responsible for training model
"""

import asyncio
import sys
import logging
import os

from typing import Optional, List, Set, Tuple

import aiofiles
from pydantic.dataclasses import dataclass
from starlette.applications import Starlette

import const as C

logger = logging.getLogger(__name__)



@dataclass
class Trial:
  id: int
  num_features: int
  gpu_idx: int
  best_acc: float = 0.0


@dataclass
class Resource:
  free: Set[int]
  busy: Set[int]

  def acquire(self) -> int:
    idx = self.free.pop()
    self.busy.add(idx)
    return idx

  def release(self, idx: int):
    self.free.add(idx)
    self.busy.remove(idx)


class Trainer:
  def __init__(self):
    self.intermediate_features = []
    self.running_models: List[Trial] = []
    self.trained_models: List[Trial] = []

    self.best_model: Optional[Tuple[Trial, str]] = None

    self.gpus = Resource(free=set(range(C.MAX_GPUS)), busy=set())
    self._schedule_event = asyncio.Event()
    self._trial_idx = 0

  async def start(self):
    asyncio.create_task(self._run_scheduler())

  async def _run_scheduler(self):
    while True:
      await self._schedule_event.wait()
      self._schedule_event.clear()

      if not self.gpus.free:
        continue
      gpu_idx = self.gpus.acquire()
      asyncio.create_task(self._launch_task(gpu_idx))

  def reschedule(self):
    self._schedule_event.set()

  def add_model(self):
    pass

  async def add_intermediate_feature(self, request):
    """Add a intermediate feature to the system."""
    filename = f"{len(self.intermediate_features)}-feature"
    fullname = os.path.join(C.FEATURE_PATH, filename)

    async with aiofiles.open(fullname, 'wb') as output:
      await output.write(await request.body())  # todo change to streaming

    self.intermediate_features.append(filename)
    self.reschedule()



  def get_best_model_file(self):
    """Return filename of the best model trained so far."""
    return self.best_model[1]

  def get_best_model_info(self):
    """Return information about the best model trained so far."""
    return self.best_model[0]

  async def _launch_task(self, gpu_idx: int):
    env = dict(CUDA_VISIBLE_DEVICES=str(gpu_idx), CALLBACK_URL="127.0.0.1:8000")
    command = C.BACKGROUND_JOB
    proc = await asyncio.create_subprocess_exec(
      sys.executable, command, env=env,
      stdout=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    self.gpus.release(gpu_idx)











_TRAINER: Optional[Trainer] = None


def register(app: Starlette) -> None:
    """Register trainer on app startup, and close no stop.

    Args:
        app (Starlette): starlette application

    """

    @app.on_event("startup")
    async def init_stub() -> None:  # pylint: disable=unused-variable
        global _TRAINER  # pylint: disable=global-statement
        _TRAINER = Trainer()
        await _TRAINER.start()
        logger.info("Trainer registered")

    @app.on_event("shutdown")
    async def close_stub() -> None:  # pylint: disable=unused-variable
        global _TRAINER  # pylint: disable=global-statement
        del _TRAINER


async def get_trainer() -> Trainer:
    """Returns a trainer."""
    global _TRAINER  # pylint: disable=global-statement
    assert _TRAINER is not None
    return _TRAINER
