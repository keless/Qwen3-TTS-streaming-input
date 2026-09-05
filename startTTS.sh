#!/bin/bash

python3 -m uvicorn tts_server:app --host 0.0.0.0 --port 8002
