@echo off
echo Starting PolicyCortex API Gateway with GPT-5/GLM-4.5...
cd backend\services\api_gateway
python -m uvicorn main:app --reload --port 8080 --host 0.0.0.0
pause