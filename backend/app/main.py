from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, universal, hard_routing, soft_moe, face_sketch

app = FastAPI(title='Restoration API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['X-Inference-Ms'],
)

app.include_router(health.router,        tags=['health'])
app.include_router(universal.router,     tags=['task1'])
app.include_router(hard_routing.router,  tags=['task2'])
app.include_router(soft_moe.router,      tags=['task3'])
app.include_router(face_sketch.router,   tags=['task4'])


@app.get('/')
def root():
    return {'service': 'Restoration API',
            'docs': '/docs',
            'health': '/health'}