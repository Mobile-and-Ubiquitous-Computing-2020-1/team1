"""Trainer component.

Responsible for training model
"""

import asyncio
import sys
import logging

from typing import Optional, List, Set

from pydantic.dataclasses import dataclass
from starlette.applications import Starlette

import const as C

logger = logging.getLogger(__name__)



class Trial:
  pass


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
    self.trained_models: List[str] = []

    self.intermediate_features = []


    self.running_models = []

    self.gpus = Resource(free=set(range(C.MAX_GPUS)), busy=set())

  def add_model(self):
    pass

  def add_intermediate_feature(self, filename: str):
    """Add a intermediate feature to the system."""
    self.intermediate_features.append(filename)
    if not self.gpus.free:
      return

    gpu_idx = self.gpus.acquire()
    asyncio.create_task(self._launch_task(gpu_idx))

  def get_best_model_file(self):
    """Return filename of the best model trained so far."""

  def get_best_model_info(self):
    """Return information about the best model trained so far."""

  async def _launch_task(self, gpu_idx: int):
    env = dict(CUDA_VISIBLE_DEVICES=str(gpu_idx))
    command = C.BACKGROUND_JOB
    proc = await asyncio.create_subprocess_exec(sys.executable, command, env=env)
    await proc.wait()











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
