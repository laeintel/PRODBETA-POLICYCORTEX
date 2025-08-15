@echo off
REM Set Azure OpenAI environment variables for PolicyCortex
REM Replace these with your actual Azure OpenAI values

REM Azure OpenAI Endpoint (from Azure AI Foundry or Azure OpenAI Service)
set AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

REM Azure OpenAI API Key
set AZURE_OPENAI_API_KEY=your-api-key-here

REM Optional: Deployment names
set AZURE_OPENAI_REALTIME_DEPLOYMENT=gpt-4
set AZURE_OPENAI_API_VERSION=2024-05-01-preview

echo Azure OpenAI environment variables set:
echo AZURE_OPENAI_ENDPOINT=%AZURE_OPENAI_ENDPOINT%
echo AZURE_OPENAI_API_KEY=****hidden****
echo AZURE_OPENAI_REALTIME_DEPLOYMENT=%AZURE_OPENAI_REALTIME_DEPLOYMENT%
echo AZURE_OPENAI_API_VERSION=%AZURE_OPENAI_API_VERSION%