#!/bin/bash
cd /home/runner/workspace/backend
exec /home/runner/workspace/.pythonlibs/bin/python3.12 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
