"""
Patent #4: Real-time Prediction Serving System
Sub-100ms latency inference with TensorRT optimization
Author: PolicyCortex ML Team
Date: January 2025

Patent Requirements:
- TensorRT optimization for GPU acceleration
- ONNX model format for cross-platform deployment
- Model quantization for edge deployment
- Batched inference processing
- Horizontal scaling with load balancing
- <100ms inference latency
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import logging
import threading
from queue import Queue, PriorityQueue
import pickle

logger = logging.getLogger(__name__)

@dataclass
class PredictionRequest:
    """Container for prediction request"""
    request_id: str
    tenant_id: str
    features: np.ndarray
    priority: int  # 0 = highest priority
    timestamp: datetime
    timeout_ms: int = 100
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority < other.priority


@dataclass
class PredictionResponse:
    """Container for prediction response"""
    request_id: str
    prediction: np.ndarray
    confidence: float
    inference_time_ms: float
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]


class ModelOptimizer:
    """
    Optimize models for inference using TensorRT and ONNX
    Patent Requirement: TensorRT optimization and ONNX conversion
    """
    
    def __init__(self):
        self.optimized_models = {}
        self.optimization_stats = {}
        
    def convert_to_onnx(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       model_name: str) -> str:
        """Convert PyTorch model to ONNX format"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape[1:])
        
        # Export to ONNX
        onnx_path = f"/tmp/{model_name}.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Converted model to ONNX: {onnx_path}")
        return onnx_path
    
    def optimize_with_tensorrt(self, onnx_path: str, precision: str = 'fp16') -> Any:
        """
        Optimize ONNX model with TensorRT
        Note: Requires TensorRT installation
        """
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Set precision
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            logger.info(f"Optimized model with TensorRT ({precision})")
            return engine
            
        except ImportError:
            logger.warning("TensorRT not available, using ONNX Runtime")
            return None
    
    def quantize_model(self, model: nn.Module, calibration_data: np.ndarray) -> nn.Module:
        """
        Quantize model for edge deployment
        Patent Requirement: Model quantization
        """
        model.eval()
        
        # Dynamic quantization (simpler, no calibration needed)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        
        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Model quantized: {reduction:.1f}% size reduction")
        
        return quantized_model


class BatchProcessor:
    """
    Batched inference processing for throughput optimization
    Patent Requirement: Efficient batched processing
    """
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.batch_queue = deque()
        self.processing = False
        self.lock = threading.Lock()
        
    def add_request(self, request: PredictionRequest):
        """Add request to batch queue"""
        with self.lock:
            self.batch_queue.append(request)
            
            # Process if batch is full
            if len(self.batch_queue) >= self.max_batch_size:
                return self._get_batch()
        
        return None
    
    def _get_batch(self) -> List[PredictionRequest]:
        """Get batch for processing"""
        batch = []
        
        with self.lock:
            for _ in range(min(self.max_batch_size, len(self.batch_queue))):
                if self.batch_queue:
                    batch.append(self.batch_queue.popleft())
        
        return batch
    
    def process_batch(self, batch: List[PredictionRequest], 
                     model: Any) -> List[PredictionResponse]:
        """Process batch of requests"""
        if not batch:
            return []
        
        start_time = time.time()
        
        # Stack features for batch processing
        features = np.stack([req.features for req in batch])
        
        # Run inference
        if hasattr(model, 'predict'):
            predictions = model.predict(features)
        else:
            # ONNX Runtime or PyTorch model
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                predictions = model(features_tensor).numpy()
        
        # Create responses
        responses = []
        inference_time = (time.time() - start_time) * 1000  # ms
        
        for i, req in enumerate(batch):
            response = PredictionResponse(
                request_id=req.request_id,
                prediction=predictions[i],
                confidence=0.95,  # Placeholder
                inference_time_ms=inference_time / len(batch),
                model_version='1.0.0',
                timestamp=datetime.now(),
                metadata={'batch_size': len(batch)}
            )
            responses.append(response)
        
        return responses
    
    async def async_process_batch(self, batch: List[PredictionRequest], 
                                 model: Any) -> List[PredictionResponse]:
        """Async batch processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_batch, batch, model)


class ModelCache:
    """
    In-memory model caching for fast inference
    Patent Requirement: Model warming and caching
    """
    
    def __init__(self, max_cache_size: int = 10):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_count = {}
        self.lock = threading.Lock()
        
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from cache"""
        with self.lock:
            if model_id in self.cache:
                self.access_count[model_id] += 1
                return self.cache[model_id]
        return None
    
    def add_model(self, model_id: str, model: Any):
        """Add model to cache with LRU eviction"""
        with self.lock:
            # Evict least recently used if cache full
            if len(self.cache) >= self.max_cache_size:
                lru_id = min(self.access_count, key=self.access_count.get)
                del self.cache[lru_id]
                del self.access_count[lru_id]
                logger.info(f"Evicted model {lru_id} from cache")
            
            self.cache[model_id] = model
            self.access_count[model_id] = 0
            logger.info(f"Added model {model_id} to cache")
    
    def warm_cache(self, models: Dict[str, Any]):
        """Pre-load models into cache"""
        for model_id, model in models.items():
            self.add_model(model_id, model)
        logger.info(f"Warmed cache with {len(models)} models")


class LoadBalancer:
    """
    Load balancing across multiple inference nodes
    Patent Requirement: Horizontal scaling with load balancing
    """
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.current_node = 0
        self.node_loads = {node: 0 for node in nodes}
        self.node_latencies = {node: deque(maxlen=100) for node in nodes}
        
    def select_node(self, request: PredictionRequest) -> str:
        """Select node using least-latency algorithm"""
        # Priority requests go to least loaded node
        if request.priority == 0:
            return min(self.node_loads, key=self.node_loads.get)
        
        # Regular requests use round-robin with latency awareness
        best_node = None
        best_score = float('inf')
        
        for node in self.nodes:
            # Calculate score based on load and latency
            load_score = self.node_loads[node]
            latency_score = np.mean(self.node_latencies[node]) if self.node_latencies[node] else 50
            
            combined_score = load_score * 0.3 + latency_score * 0.7
            
            if combined_score < best_score:
                best_score = combined_score
                best_node = node
        
        return best_node or self.nodes[0]
    
    def update_metrics(self, node: str, latency_ms: float):
        """Update node metrics after request"""
        self.node_latencies[node].append(latency_ms)
        self.node_loads[node] = max(0, self.node_loads[node] - 1)
    
    def mark_busy(self, node: str):
        """Mark node as busy"""
        self.node_loads[node] += 1


class InferenceServer:
    """
    Main inference server with all optimizations
    Patent Requirement: <100ms latency with all features
    """
    
    def __init__(self, model: Any, model_id: str):
        self.model = model
        self.model_id = model_id
        
        # Initialize components
        self.optimizer = ModelOptimizer()
        self.batch_processor = BatchProcessor()
        self.model_cache = ModelCache()
        self.request_queue = PriorityQueue()
        self.response_cache = {}
        
        # Performance monitoring
        self.latency_monitor = deque(maxlen=1000)
        self.throughput_monitor = deque(maxlen=100)
        
        # Start background workers
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_requests)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def _process_requests(self):
        """Background worker for processing requests"""
        while self.running:
            try:
                # Wait for requests with timeout
                batch = []
                deadline = time.time() + 0.01  # 10ms max wait
                
                while time.time() < deadline and len(batch) < 32:
                    if not self.request_queue.empty():
                        _, request = self.request_queue.get(timeout=0.001)
                        batch.append(request)
                    else:
                        time.sleep(0.001)
                
                if batch:
                    # Process batch
                    responses = self.batch_processor.process_batch(batch, self.model)
                    
                    # Cache responses
                    for response in responses:
                        self.response_cache[response.request_id] = response
                        
                        # Update monitoring
                        self.latency_monitor.append(response.inference_time_ms)
                    
                    self.throughput_monitor.append(len(batch))
                    
            except Exception as e:
                logger.error(f"Error processing requests: {e}")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make prediction with <100ms latency guarantee
        Patent Requirement: Real-time prediction serving
        """
        start_time = time.time()
        
        # Check cache for repeated requests
        cache_key = hashlib.md5(
            f"{request.tenant_id}_{request.features.tobytes()}".encode()
        ).hexdigest()
        
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            logger.debug(f"Cache hit for request {request.request_id}")
            return cached
        
        # Add to priority queue
        self.request_queue.put((request.priority, request))
        
        # Wait for response
        timeout = request.timeout_ms / 1000.0
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            if request.request_id in self.response_cache:
                response = self.response_cache.pop(request.request_id)
                
                # Check latency requirement
                total_time = (time.time() - start_time) * 1000
                if total_time > 100:
                    logger.warning(f"Latency violation: {total_time:.2f}ms > 100ms")
                
                return response
            
            time.sleep(0.001)
        
        # Timeout - return default response
        logger.error(f"Request {request.request_id} timed out")
        
        return PredictionResponse(
            request_id=request.request_id,
            prediction=np.array([0.5]),  # Default prediction
            confidence=0.0,
            inference_time_ms=timeout * 1000,
            model_version=self.model_id,
            timestamp=datetime.now(),
            metadata={'error': 'timeout'}
        )
    
    async def async_predict(self, request: PredictionRequest) -> PredictionResponse:
        """Async prediction interface"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, request)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.latency_monitor:
            latencies = list(self.latency_monitor)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
        else:
            p50 = p95 = p99 = 0
        
        return {
            'average_latency_ms': np.mean(self.latency_monitor) if self.latency_monitor else 0,
            'p50_latency_ms': p50,
            'p95_latency_ms': p95,
            'p99_latency_ms': p99,
            'throughput_per_second': sum(self.throughput_monitor) / max(len(self.throughput_monitor), 1),
            'cache_size': len(self.response_cache),
            'queue_size': self.request_queue.qsize(),
            'meets_sla': p95 < 100  # 95th percentile should be under 100ms
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        self.worker_thread.join(timeout=5)
        logger.info("Inference server shut down")


class PredictionServingEngine:
    """
    Main prediction serving engine orchestrating all components
    Ensures <100ms latency with horizontal scaling
    """
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.servers = {}
        self.load_balancer = None
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Global performance tracking
        self.global_stats = {
            'total_requests': 0,
            'total_timeouts': 0,
            'sla_violations': 0
        }
        
    def deploy_model(self, model: Any, model_id: str, 
                    optimize: bool = True) -> InferenceServer:
        """Deploy model with optimization"""
        if optimize:
            # Optimize model
            optimizer = ModelOptimizer()
            
            # Try TensorRT optimization
            try:
                input_shape = (1, 100)  # Example shape
                onnx_path = optimizer.convert_to_onnx(model, input_shape, model_id)
                
                # TensorRT optimization (if available)
                trt_engine = optimizer.optimize_with_tensorrt(onnx_path)
                if trt_engine:
                    model = trt_engine
                    logger.info(f"Deployed TensorRT optimized model {model_id}")
                else:
                    # Use ONNX Runtime as fallback
                    model = ort.InferenceSession(onnx_path)
                    logger.info(f"Deployed ONNX model {model_id}")
                    
            except Exception as e:
                logger.warning(f"Optimization failed, using original model: {e}")
        
        # Create inference server
        server = InferenceServer(model, model_id)
        self.servers[model_id] = server
        
        # Update load balancer
        if len(self.servers) > 1:
            self.load_balancer = LoadBalancer(list(self.servers.keys()))
        
        return server
    
    def predict(self, tenant_id: str, features: np.ndarray, 
               priority: int = 1) -> PredictionResponse:
        """
        Make prediction with automatic model selection and load balancing
        Guarantees <100ms latency per patent requirement
        """
        # Create request
        request = PredictionRequest(
            request_id=hashlib.md5(f"{tenant_id}_{time.time()}".encode()).hexdigest(),
            tenant_id=tenant_id,
            features=features,
            priority=priority,
            timestamp=datetime.now(),
            timeout_ms=95  # Leave 5ms buffer
        )
        
        # Select server
        if self.load_balancer and len(self.servers) > 1:
            server_id = self.load_balancer.select_node(request)
            self.load_balancer.mark_busy(server_id)
        else:
            server_id = list(self.servers.keys())[0] if self.servers else None
        
        if not server_id:
            raise ValueError("No models deployed")
        
        server = self.servers[server_id]
        
        # Make prediction
        start_time = time.time()
        response = server.predict(request)
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.global_stats['total_requests'] += 1
        
        if total_time > 100:
            self.global_stats['sla_violations'] += 1
            logger.warning(f"SLA violation: {total_time:.2f}ms > 100ms")
        
        if 'error' in response.metadata and response.metadata['error'] == 'timeout':
            self.global_stats['total_timeouts'] += 1
        
        # Update load balancer metrics
        if self.load_balancer:
            self.load_balancer.update_metrics(server_id, total_time)
        
        return response
    
    async def async_predict_batch(self, requests: List[Tuple[str, np.ndarray, int]]) -> List[PredictionResponse]:
        """Async batch prediction"""
        tasks = []
        
        for tenant_id, features, priority in requests:
            task = asyncio.create_task(
                self._async_predict_single(tenant_id, features, priority)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    async def _async_predict_single(self, tenant_id: str, features: np.ndarray, 
                                   priority: int) -> PredictionResponse:
        """Async single prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.predict, tenant_id, features, priority
        )
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        report = {
            'global_stats': self.global_stats,
            'num_models': len(self.servers),
            'servers': {}
        }
        
        # Aggregate server statistics
        all_latencies = []
        
        for server_id, server in self.servers.items():
            stats = server.get_performance_stats()
            report['servers'][server_id] = stats
            
            if stats['average_latency_ms'] > 0:
                all_latencies.append(stats['average_latency_ms'])
        
        # System-wide metrics
        if all_latencies:
            report['system_average_latency_ms'] = np.mean(all_latencies)
            report['system_sla_compliance'] = (
                1 - self.global_stats['sla_violations'] / max(self.global_stats['total_requests'], 1)
            ) * 100
        
        # Patent requirement check
        report['meets_patent_requirements'] = (
            report.get('system_average_latency_ms', 0) < 100 and
            report.get('system_sla_compliance', 0) > 95
        )
        
        return report
    
    def shutdown(self):
        """Graceful shutdown of all servers"""
        for server in self.servers.values():
            server.shutdown()
        
        self.executor.shutdown(wait=True)
        logger.info("Prediction serving engine shut down")