# ML Docker Infrastructure Troubleshooting Guide

## Overview
This guide addresses common issues when running ML Docker containers on Windows without GPU support.

## Quick Start Commands

### Build the ML Docker Image (CPU-only)
```bash
docker build -f Dockerfile.ml-cpu -t policycortex-ml-cpu:latest .
```

### Run Quick Test
```bash
.\scripts\quick-ml-test.bat
```

### Run Full Test Suite
```bash
.\scripts\test-ml-docker.bat
```

### Start All Services
```bash
docker-compose -f docker-compose.ml-windows.yml up -d
```

### Stop All Services
```bash
docker-compose -f docker-compose.ml-windows.yml down
```

### View Logs
```bash
docker-compose -f docker-compose.ml-windows.yml logs -f
```

## Common Issues and Solutions

### 1. Docker Build Failures

#### Issue: "Package not found" during pip install
**Solution:**
- Check internet connectivity
- Try building with `--no-cache` flag:
  ```bash
  docker build --no-cache -f Dockerfile.ml-cpu -t policycortex-ml-cpu:latest .
  ```

#### Issue: "Out of space" error
**Solution:**
- Clean up Docker system:
  ```bash
  docker system prune -a
  ```
- Check available disk space

### 2. Service Startup Issues

#### Issue: "Port already in use"
**Solution:**
- Check which process is using the port:
  ```bash
  netstat -ano | findstr :8080
  netstat -ano | findstr :8765
  netstat -ano | findstr :5432
  ```
- Stop conflicting services or change ports in docker-compose.ml-windows.yml

#### Issue: Services fail health checks
**Solution:**
- Increase startup timeout in docker-compose.ml-windows.yml:
  ```yaml
  healthcheck:
    start_period: 120s  # Increase from 90s
  ```
- Check service logs:
  ```bash
  docker logs ml-prediction-server
  docker logs ml-websocket-server
  ```

### 3. Database Connection Issues

#### Issue: "Connection refused" to PostgreSQL
**Solution:**
- Ensure PostgreSQL is healthy:
  ```bash
  docker exec postgres-ml pg_isready -U postgres
  ```
- Check if database exists:
  ```bash
  docker exec postgres-ml psql -U postgres -c "\l"
  ```
- Recreate database:
  ```bash
  docker exec postgres-ml psql -U postgres -c "CREATE DATABASE policycortex;"
  docker exec postgres-ml psql -U postgres -c "CREATE DATABASE mlflow;"
  ```

### 4. ML Model Issues

#### Issue: "CUDA not available" warnings
**Solution:**
This is expected on Windows without GPU. The services automatically fall back to CPU mode.
Set environment variable to suppress warnings:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=""
  - CPU_ONLY=true
```

#### Issue: Slow inference times
**Solution:**
- Reduce batch size in environment variables:
  ```yaml
  - INFERENCE_BATCH_SIZE=4  # Reduce from 8
  ```
- Increase CPU resources in Docker Desktop settings
- Use model quantization (if implemented)

### 5. Memory Issues

#### Issue: Container killed due to OOM (Out of Memory)
**Solution:**
- Increase Docker Desktop memory allocation (Settings > Resources)
- Reduce model cache size:
  ```yaml
  - MODEL_CACHE_SIZE=3  # Reduce from 5
  ```
- Limit Redis memory usage (already configured in docker-compose)

### 6. Network Issues

#### Issue: Services cannot communicate
**Solution:**
- Verify network exists:
  ```bash
  docker network ls | findstr policycortex
  ```
- Recreate network:
  ```bash
  docker network create policycortex-network
  ```
- Use container names for internal communication (e.g., `postgres-ml` instead of `localhost`)

## Windows-Specific Considerations

### File Path Issues
- Use forward slashes in Docker volumes: `./backend/migrations` not `.\backend\migrations`
- Ensure line endings are LF not CRLF for shell scripts

### Permission Issues
- Run Docker Desktop as Administrator if permission errors occur
- Ensure current user is in the docker-users group

### WSL2 vs Hyper-V
- WSL2 backend (recommended): Better performance, Linux compatibility
- Hyper-V backend: Legacy, may have compatibility issues

## Performance Optimization

### CPU-Only Optimizations
1. **Use MKL optimizations**:
   ```dockerfile
   ENV MKL_NUM_THREADS=4
   ENV OMP_NUM_THREADS=4
   ```

2. **Enable ONNX Runtime for inference**:
   Already included in requirements-ml-cpu.txt

3. **Reduce model complexity**:
   - Use smaller models for testing
   - Implement model pruning
   - Use quantization

### Docker Settings
1. Increase CPU allocation in Docker Desktop
2. Allocate at least 8GB RAM
3. Use WSL2 backend for better performance

## Monitoring and Debugging

### Check Resource Usage
```bash
docker stats
```

### View Real-time Logs
```bash
docker-compose -f docker-compose.ml-windows.yml logs -f ml-prediction-server
```

### Access Container Shell
```bash
docker exec -it ml-prediction-server /bin/bash
```

### Test ML Models Manually
```bash
docker exec ml-prediction-server python3 -c "
from ml_models.ensemble_engine import EnsembleEngine
engine = EnsembleEngine()
print('Model loaded successfully')
"
```

## Health Check Endpoints

- **ML Prediction Server**: http://localhost:8080/health
- **MLflow**: http://localhost:5000/health
- **WebSocket**: ws://localhost:8765 (connection test)
- **Metrics**: http://localhost:9090/metrics

## Cleanup Commands

### Stop and Remove All Containers
```bash
docker-compose -f docker-compose.ml-windows.yml down -v
```

### Remove All ML-related Images
```bash
docker rmi policycortex-ml-cpu:latest
docker rmi policycortex-ml-cpu:test
```

### Complete Cleanup
```bash
docker system prune -a --volumes
```

## Getting Help

1. Check service logs first:
   ```bash
   docker-compose -f docker-compose.ml-windows.yml logs [service-name]
   ```

2. Verify Docker and system resources:
   ```bash
   docker version
   docker system df
   ```

3. Test individual components:
   ```bash
   .\scripts\quick-ml-test.bat
   ```

4. For persistent issues, collect diagnostics:
   ```bash
   docker-compose -f docker-compose.ml-windows.yml ps
   docker-compose -f docker-compose.ml-windows.yml logs --tail=100 > ml-logs.txt
   ```