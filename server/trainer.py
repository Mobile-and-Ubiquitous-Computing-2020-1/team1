"""Trainer component.

Responsible for training model
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

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
  model_path: str = ""


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
    self.running_models: Dict[int, Trial] = {}
    self.trained_models: List[Trial] = []

    self.best_model: Optional[Trial] = None

    self.gpus = Resource(free=set(range(C.MAX_GPUS)), busy=set())
    self._schedule_event = asyncio.Event()
    self._trial_idx = 0

    self._max_feature_size = 0

  async def start(self):
    asyncio.create_task(self._run_scheduler())

  async def _run_scheduler(self):
    while True:
      await self._schedule_event.wait()
      self._schedule_event.clear()

      if not self.gpus.free:
        continue

      if self._max_feature_size >= len(self.intermediate_features):
        continue

      gpu_idx = self.gpus.acquire()
      self._max_feature_size = len(self.intermediate_features)
      trial_id = self._trial_idx
      self._trial_idx += 1
      asyncio.create_task(self._launch_task(trial_id, gpu_idx))

  def reschedule(self):
    self._schedule_event.set()

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
    assert self.best_model
    return self.best_model.model_path

  def get_best_model_info(self):
    """Return information about the best model trained so far."""
    assert self.best_model
    return self.best_model

  def update_model(self, acc):
    trial_id = acc.trial_id
    best_acc = acc.acc
    trial = self.running_models[trial_id]
    trial.best_acc = best_acc

  async def _launch_task(self, trial_id: int, gpu_idx: int):
    trial = Trial(trial_id, len(self.intermediate_features), gpu_idx, 0.0)

    model_path = f"{trial.id}.tflite"
    trial.model_path = model_path
    self.running_models[trial.id] = trial

    env = dict(
      CUDA_VISIBLE_DEVICES=str(gpu_idx),
      CALLBACK_URL="127.0.0.1:8000",
      MODEL_PATH=model_path,
      TRIAL_ID=str(trial_id)
    )
    command = C.BACKGROUND_JOB
    proc = await asyncio.create_subprocess_exec(
      sys.executable, command, env=env,
      stdout=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    self.gpus.release(gpu_idx)
    self.trained_models.append(trial)

    if not self.best_model:
      self.best_model = trial
    elif trial.best_acc > self.best_model.best_acc:
      self.best_model = trial
    self.reschedule()










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
