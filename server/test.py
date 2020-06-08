"""Test code for server."""

import pytest

from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
  return TestClient(app)


def test_ping(client: TestClient):
  response = client.get("/ping")
  assert response.status_code == 200
  assert response.text == "pong"


def test_push(client: TestClient):
  data = b"example data"
  filename = "example.o"

  headers = {
    'Content-Type': 'application/octet-stream',
    'Content-Disposition': f'filename={filename}',
  }

  response = client.post("/push", data=data, headers=headers)
  assert response.status_code == 200
  assert response.json() == dict(success=True)


# def test_pull(client: TestClient):


#   response = client.get("/pull/")
