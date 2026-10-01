"""Run: uvicorn main:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import torch

from recommender import DraftRequest, Recommender


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(2)
        app.state.recommender = None
        app.state.model_error = 'Train a partial-draft model, set MODEL_PATH, then restart the API.'
        path = Path(model_path or os.getenv('MODEL_PATH', 'artifacts/draft/model.pt'))
        try:
            app.state.recommender = Recommender(path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logging.getLogger(__name__).warning('Model unavailable (%s). %s', type(exc).__name__, app.state.model_error)
        yield
        app.state.recommender = None

    app = FastAPI(title='League Draft Recommender', version='0.1.0', lifespan=lifespan)

    def model():
        if app.state.recommender is None:
            raise HTTPException(status_code=503, detail=app.state.model_error)
        return app.state.recommender

    @app.get('/health')
    def health():
        recommender = model()
        return {'status': 'ready', 'training_matches': recommender.training_matches,
                'supported_champions': len(recommender.supported)}

    @app.get('/champions')
    def champions():
        return model().champions()

    @app.post('/recommendations')
    def recommendations(draft: DraftRequest):
        try:
            return model().recommend(draft)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
