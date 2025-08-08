"""
Pydantic models for AI Engine service.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Model status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class AnalysisType(str, Enum):
    """Analysis type enumeration."""

    COMPLIANCE = "compliance"
    SECURITY = "security"
    COST = "cost"
    PERFORMANCE = "performance"
    GOVERNANCE = "governance"


class DetectionType(str, Enum):
    """Anomaly detection type enumeration."""

    RESOURCE_USAGE = "resource_usage"
    COST_ANOMALY = "cost_anomaly"
    SECURITY_ANOMALY = "security_anomaly"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    COMPLIANCE_ANOMALY = "compliance_anomaly"


class OptimizationGoal(str, Enum):
    """Cost optimization goal enumeration."""

    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    BALANCE_COST_PERFORMANCE = "balance_cost_performance"
    OPTIMIZE_UTILIZATION = "optimize_utilization"


class SentimentType(str, Enum):
    """Sentiment analysis type enumeration."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ModelType(str, Enum):
    """Model type enumeration."""

    NLP = "nlp"
    ANOMALY_DETECTION = "anomaly_detection"
    COST_OPTIMIZATION = "cost_optimization"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class APIResponse(BaseModel):
    """Generic API response model."""

    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ModelInfo(BaseModel):
    """Model information model."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: ModelType = Field(..., description="Model type")
    status: ModelStatus = Field(..., description="Model status")
    description: Optional[str] = Field(None, description="Model description")
    created_at: datetime = Field(..., description="Model creation timestamp")
    updated_at: datetime = Field(..., description="Model last update timestamp")
    size_mb: Optional[float] = Field(None, description="Model size in MB")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")
    training_data_info: Optional[Dict[str, Any]] = Field(
        None, description="Training data information"
    )


class ModelTrainingRequest(BaseModel):
    """Model training request model."""

    training_data: Dict[str, Any] = Field(..., description="Training data configuration")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Training parameters")
    validation_split: float = Field(0.2, description="Validation data split ratio")
    epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.001, description="Learning rate")
    early_stopping: bool = Field(True, description="Enable early stopping")


class ModelTrainingResponse(BaseModel):
    """Model training response model."""

    task_id: str = Field(..., description="Training task identifier")
    model_name: str = Field(..., description="Model name")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Training message")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Training start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class ModelMetrics(BaseModel):
    """Model performance metrics model."""

    model_name: str = Field(..., description="Model name")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    inference_time_ms: float = Field(..., description="Average inference time in milliseconds")
    total_predictions: int = Field(..., description="Total number of predictions made")
    drift_score: Optional[float] = Field(None, description="Model drift score")
    last_updated: datetime = Field(..., description="Last metrics update timestamp")
    additional_metrics: Optional[Dict[str, float]] = Field(None, description="Additional metrics")


class PolicyAnalysisRequest(BaseModel):
    """Policy analysis request model."""

    request_id: str = Field(..., description="Request identifier")
    policy_text: str = Field(..., description="Policy document text")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    options: Optional[Dict[str, Any]] = Field(None, description="Analysis options")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class PolicyAnalysisResponse(BaseModel):
    """Policy analysis response model."""

    request_id: str = Field(..., description="Request identifier")
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    confidence_score: float = Field(..., description="Confidence score")
    key_insights: Optional[List[str]] = Field(None, description="Key insights")
    recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="Recommendations")
    compliance_status: Optional[Dict[str, Any]] = Field(None, description="Compliance status")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request model."""

    request_id: str = Field(..., description="Request identifier")
    resource_data: Dict[str, Any] = Field(..., description="Resource data for analysis")
    detection_type: DetectionType = Field(..., description="Type of anomaly detection")
    threshold: float = Field(0.95, description="Anomaly detection threshold")
    time_window: Optional[str] = Field(None, description="Time window for analysis")
    baseline_data: Optional[Dict[str, Any]] = Field(
        None, description="Baseline data for comparison"
    )


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response model."""

    request_id: str = Field(..., description="Request identifier")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    analysis_summary: Dict[str, Any] = Field(..., description="Analysis summary")
    confidence_score: float = Field(..., description="Overall confidence score")
    severity_levels: Optional[Dict[str, int]] = Field(None, description="Anomaly severity counts")
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Remediation recommendations"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class CostOptimizationRequest(BaseModel):
    """Cost optimization request model."""

    request_id: str = Field(..., description="Request identifier")
    resource_data: Dict[str, Any] = Field(..., description="Resource data for optimization")
    optimization_goals: List[OptimizationGoal] = Field(..., description="Optimization goals")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")
    budget_limits: Optional[Dict[str, float]] = Field(None, description="Budget limitations")
    time_horizon: Optional[str] = Field(None, description="Optimization time horizon")


class CostOptimizationResponse(BaseModel):
    """Cost optimization response model."""

    request_id: str = Field(..., description="Request identifier")
    recommendations: List[Dict[str, Any]] = Field(
        ..., description="Cost optimization recommendations"
    )
    projected_savings: Dict[str, float] = Field(..., description="Projected cost savings")
    implementation_plan: Dict[str, Any] = Field(..., description="Implementation plan")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment")
    confidence_score: float = Field(..., description="Confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class PredictiveAnalyticsRequest(BaseModel):
    """Predictive analytics request model."""

    request_id: str = Field(..., description="Request identifier")
    historical_data: Dict[str, Any] = Field(..., description="Historical data for analysis")
    prediction_horizon: str = Field(..., description="Prediction time horizon")
    metrics: List[str] = Field(..., description="Metrics to predict")
    seasonality: Optional[bool] = Field(None, description="Consider seasonality patterns")
    external_factors: Optional[Dict[str, Any]] = Field(
        None, description="External factors to consider"
    )


class PredictiveAnalyticsResponse(BaseModel):
    """Predictive analytics response model."""

    request_id: str = Field(..., description="Request identifier")
    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    trends: Dict[str, Any] = Field(..., description="Trend analysis")
    forecast_accuracy: Dict[str, float] = Field(..., description="Forecast accuracy metrics")
    confidence_intervals: Dict[str, Dict[str, float]] = Field(
        ..., description="Confidence intervals"
    )
    risk_factors: Optional[List[Dict[str, Any]]] = Field(None, description="Risk factors")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SentimentAnalysisRequest(BaseModel):
    """Sentiment analysis request model."""

    request_id: str = Field(..., description="Request identifier")
    text: str = Field(..., description="Text to analyze")
    analysis_type: Optional[str] = Field(None, description="Type of sentiment analysis")
    language: str = Field("en", description="Text language")
    options: Optional[Dict[str, Any]] = Field(None, description="Analysis options")


class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response model."""

    request_id: str = Field(..., description="Request identifier")
    sentiment: SentimentType = Field(..., description="Overall sentiment")
    confidence_score: float = Field(..., description="Confidence score")
    emotions: Dict[str, float] = Field(..., description="Emotion scores")
    key_phrases: List[str] = Field(..., description="Key phrases")
    sentiment_scores: Optional[Dict[str, float]] = Field(
        None, description="Detailed sentiment scores"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class FeatureEngineeringRequest(BaseModel):
    """Feature engineering request model."""

    request_id: str = Field(..., description="Request identifier")
    raw_data: Dict[str, Any] = Field(..., description="Raw data for feature engineering")
    feature_types: List[str] = Field(..., description="Types of features to generate")
    preprocessing_steps: Optional[List[str]] = Field(None, description="Preprocessing steps")
    target_variable: Optional[str] = Field(
        None, description="Target variable for supervised learning"
    )


class FeatureEngineeringResponse(BaseModel):
    """Feature engineering response model."""

    request_id: str = Field(..., description="Request identifier")
    engineered_features: Dict[str, Any] = Field(..., description="Engineered features")
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores"
    )
    preprocessing_summary: Dict[str, Any] = Field(..., description="Preprocessing summary")
    feature_statistics: Dict[str, Any] = Field(..., description="Feature statistics")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelDriftRequest(BaseModel):
    """Model drift detection request model."""

    request_id: str = Field(..., description="Request identifier")
    model_name: str = Field(..., description="Model name")
    reference_data: Dict[str, Any] = Field(..., description="Reference data for comparison")
    current_data: Dict[str, Any] = Field(..., description="Current data")
    drift_threshold: float = Field(0.1, description="Drift detection threshold")


class ModelDriftResponse(BaseModel):
    """Model drift detection response model."""

    request_id: str = Field(..., description="Request identifier")
    model_name: str = Field(..., description="Model name")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Drift score")
    drift_details: Dict[str, Any] = Field(..., description="Detailed drift analysis")
    recommendations: List[str] = Field(..., description="Recommendations for handling drift")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""

    request_id: str = Field(..., description="Request identifier")
    model_name: str = Field(..., description="Model name")
    input_data: List[Dict[str, Any]] = Field(..., description="Input data for predictions")
    output_format: str = Field("json", description="Output format")
    include_confidence: bool = Field(True, description="Include confidence scores")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    request_id: str = Field(..., description="Request identifier")
    model_name: str = Field(..., description="Model name")
    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    batch_size: int = Field(..., description="Batch size processed")
    success_count: int = Field(..., description="Number of successful predictions")
    error_count: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelExplainabilityRequest(BaseModel):
    """Model explainability request model."""

    request_id: str = Field(..., description="Request identifier")
    model_name: str = Field(..., description="Model name")
    input_data: Dict[str, Any] = Field(..., description="Input data for explanation")
    explanation_type: str = Field("feature_importance", description="Type of explanation")
    top_k: int = Field(10, description="Number of top features to explain")


class ModelExplainabilityResponse(BaseModel):
    """Model explainability response model."""

    request_id: str = Field(..., description="Request identifier")
    model_name: str = Field(..., description="Model name")
    prediction: Dict[str, Any] = Field(..., description="Model prediction")
    explanations: Dict[str, Any] = Field(..., description="Explanation results")
    feature_contributions: Dict[str, float] = Field(..., description="Feature contributions")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="Visualization data")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class DataQualityRequest(BaseModel):
    """Data quality assessment request model."""

    request_id: str = Field(..., description="Request identifier")
    data: Dict[str, Any] = Field(..., description="Data for quality assessment")
    quality_checks: List[str] = Field(..., description="Quality checks to perform")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Quality thresholds")


class DataQualityResponse(BaseModel):
    """Data quality assessment response model."""

    request_id: str = Field(..., description="Request identifier")
    quality_score: float = Field(..., description="Overall quality score")
    quality_report: Dict[str, Any] = Field(..., description="Detailed quality report")
    issues_found: List[Dict[str, Any]] = Field(..., description="Quality issues found")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class AutoMLRequest(BaseModel):
    """AutoML request model."""

    request_id: str = Field(..., description="Request identifier")
    dataset: Dict[str, Any] = Field(..., description="Dataset for AutoML")
    problem_type: str = Field(..., description="Problem type (classification, regression, etc.)")
    target_column: str = Field(..., description="Target column name")
    time_budget: int = Field(3600, description="Time budget in seconds")
    metric: str = Field("accuracy", description="Optimization metric")
    validation_strategy: str = Field("cross_validation", description="Validation strategy")


class AutoMLResponse(BaseModel):
    """AutoML response model."""

    request_id: str = Field(..., description="Request identifier")
    task_id: str = Field(..., description="AutoML task identifier")
    status: str = Field(..., description="AutoML task status")
    best_model: Optional[Dict[str, Any]] = Field(None, description="Best model information")
    leaderboard: Optional[List[Dict[str, Any]]] = Field(None, description="Model leaderboard")
    experiment_summary: Dict[str, Any] = Field(..., description="Experiment summary")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelVersionRequest(BaseModel):
    """Model version request model."""

    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: Optional[str] = Field(None, description="Version description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Version metadata")


class ModelVersionResponse(BaseModel):
    """Model version response model."""

    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    status: str = Field(..., description="Version status")
    created_at: datetime = Field(..., description="Version creation timestamp")
    size_mb: Optional[float] = Field(None, description="Version size in MB")
    checksum: Optional[str] = Field(None, description="Model checksum")
    metrics: Optional[Dict[str, float]] = Field(None, description="Version metrics")


# Patent Implementation Models


class UnifiedAIAnalysisRequest(BaseModel):
    """Unified AI Platform analysis request model (Patent 2)."""

    request_id: str = Field(..., description="Request identifier")
    governance_data: Dict[str, Any] = Field(..., description="Multi-domain governance data")
    analysis_scope: List[str] = Field(..., description="Domains to analyze")
    optimization_preferences: Optional[Dict[str, float]] = Field(
        None, description="Optimization preferences"
    )


class UnifiedAIAnalysisResponse(BaseModel):
    """Unified AI Platform analysis response model (Patent 2)."""

    request_id: str = Field(..., description="Request identifier")
    optimization_scores: List[float] = Field(..., description="Multi-objective optimization scores")
    domain_correlations: Dict[str, float] = Field(..., description="Cross-domain correlations")
    embeddings: Dict[str, List[float]] = Field(..., description="Hierarchical embeddings")
    recommendations: List[Dict[str, Any]] = Field(..., description="Governance recommendations")
    confidence_score: float = Field(..., description="Analysis confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class GovernanceOptimizationRequest(BaseModel):
    """Governance optimization request model (Patent 2)."""

    request_id: str = Field(..., description="Request identifier")
    governance_data: Dict[str, Any] = Field(..., description="Current governance state")
    preferences: Dict[str, float] = Field(..., description="Optimization preferences")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")
    max_generations: Optional[int] = Field(200, description="Maximum optimization generations")


class GovernanceOptimizationResponse(BaseModel):
    """Governance optimization response model (Patent 2)."""

    request_id: str = Field(..., description="Request identifier")
    best_solution: Dict[str, Any] = Field(..., description="Best optimization solution")
    pareto_front_size: int = Field(..., description="Size of Pareto front")
    convergence_achieved: bool = Field(..., description="Whether optimization converged")
    recommendations: List[Dict[str, Any]] = Field(..., description="Implementation recommendations")
    utility_score: float = Field(..., description="Solution utility score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ConversationRequest(BaseModel):
    """Conversational governance intelligence request model (Patent 3)."""

    user_input: str = Field(..., description="Natural language user input")
    session_id: str = Field(..., description="Conversation session identifier")
    user_id: str = Field(..., description="User identifier")
    context_override: Optional[Dict[str, Any]] = Field(None, description="Context override data")


class ConversationResponse(BaseModel):
    """Conversational governance intelligence response model (Patent 3)."""

    response: str = Field(..., description="Natural language response")
    intent: str = Field(..., description="Detected user intent")
    entities: Dict[str, List[str]] = Field(..., description="Extracted entities")
    confidence: float = Field(..., description="Intent classification confidence")
    api_call: Optional[Dict[str, Any]] = Field(None, description="Generated API call")
    clarification_needed: Optional[List[str]] = Field(None, description="Required clarifications")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    success: bool = Field(..., description="Request success status")


class PolicySynthesisRequest(BaseModel):
    """Policy synthesis request model (Patent 3)."""

    request_id: str = Field(..., description="Request identifier")
    description: str = Field(..., description="Natural language policy description")
    domain: str = Field(..., description="Governance domain")
    policy_type: Optional[str] = Field("general", description="Policy type")
    constraints: Optional[List[str]] = Field(None, description="Policy constraints")


class PolicySynthesisResponse(BaseModel):
    """Policy synthesis response model (Patent 3)."""

    request_id: str = Field(..., description="Request identifier")
    policy_text: str = Field(..., description="Generated policy text")
    structured_policy: Dict[str, Any] = Field(..., description="Structured policy components")
    domain: str = Field(..., description="Policy domain")
    confidence_score: float = Field(..., description="Generation confidence")
    validation_results: Optional[Dict[str, Any]] = Field(
        None, description="Policy validation results"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ConversationHistoryRequest(BaseModel):
    """Conversation history request model."""

    session_id: str = Field(..., description="Session identifier")
    include_metadata: bool = Field(True, description="Include conversation metadata")


class ConversationHistoryResponse(BaseModel):
    """Conversation history response model."""

    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    history: List[Dict[str, Any]] = Field(..., description="Conversation history")
    current_state: str = Field(..., description="Current conversation state")
    entities: Dict[str, Any] = Field(..., description="Accumulated entities")
    success: bool = Field(..., description="Request success status")
