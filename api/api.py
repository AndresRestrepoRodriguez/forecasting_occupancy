from typing import Any

from fastapi import APIRouter
from loguru import logger
from api import __version__, schemas
from api.config import settings
from api.backend.utils import (
    generate_timestamp,
    format_predictions
)
from src.utils import create_datetime_range


from src.model import Model

DEVICES = [f'device_{num}' for num in range(1,8)]
folder = 'models/'
models = {device: Model(device) for device in DEVICES}

def load_models(models):
    for k,v in models.items():
        v.load_model(folder)


load_models(models)

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__
    )

    return health.dict()


@api_router.get("/predict", status_code=200)
async def predict() -> dict:
    predictions = []
    timestamp = generate_timestamp()
    test_data = create_datetime_range(timestamp)
    for device in DEVICES:
        results = models[device].predict_service(test_data)
        results_format = format_predictions(results, device)
        predictions.append(results_format)
    return {'results': predictions}



