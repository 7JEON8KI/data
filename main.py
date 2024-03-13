# -*- coding: utf-8 -*-

from fastapi import FastAPI
from app import routes
import uvicorn
from starlette.middleware.cors import CORSMiddleware

""" 
    fileName      : main.py
    author        : 이소민
""" 

app = FastAPI()
app.include_router(routes.router)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
