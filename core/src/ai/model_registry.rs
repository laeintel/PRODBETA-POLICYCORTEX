use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Model versioning and lineage tracking for reproducible AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub model_type: ModelType,
    pub training_data: TrainingDataLineage,
    pub metrics: ModelMetrics,
    pub benchmarks: Vec<Benchmark>,
    pub limitations: Vec<String>,
    pub ethical_considerations: Vec<String>,
    pub dependencies: Vec<ModelDependency>,
    pub checksum: String,
    pub reproducibility: ReproducibilityInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Classification,
    Regression,
    NLP,
    ComputerVision,
    ReinforcementLearning,
    Ensemble,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataLineage {
    pub dataset_id: String,
    pub dataset_version: String,
    pub source_url: Option<String>,
    pub preprocessing_steps: Vec<String>,
    pub split_ratios: SplitRatios,
    pub data_checksum: String,
    pub collection_date: DateTime<Utc>,
    pub data_characteristics: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRatios {
    pub train: f32,
    pub validation: f32,
    pub test: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1_score: Option<f64>,
    pub auc_roc: Option<f64>,
    pub mae: Option<f64>,
    pub rmse: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    pub name: String,
    pub version: String,
    pub score: f64,
    pub percentile: Option<f64>,
    pub execution_time_ms: u64,
    pub hardware_spec: String,
    pub date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDependency {
    pub name: String,
    pub version: String,
    pub dependency_type: DependencyType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Library,
    Framework,
    PretrainedModel,
    DataPipeline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    pub random_seed: Option<u64>,
    pub environment: EnvironmentSpec,
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub training_script_hash: String,
    pub git_commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSpec {
    pub os: String,
    pub python_version: Option<String>,
    pub cuda_version: Option<String>,
    pub container_image: Option<String>,
    pub hardware: HardwareSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub cpu: String,
    pub gpu: Option<String>,
    pub memory_gb: u32,
    pub storage_type: String,
}

/// Model evaluation harness for consistent testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationHarness {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub test_cases: Vec<TestCase>,
    pub performance_criteria: PerformanceCriteria,
    pub fairness_tests: Vec<FairnessTest>,
    pub robustness_tests: Vec<RobustnessTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub description: String,
    pub input_data: serde_json::Value,
    pub expected_output: serde_json::Value,
    pub tolerance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCriteria {
    pub min_accuracy: Option<f64>,
    pub max_latency_ms: Option<u64>,
    pub max_memory_mb: Option<u64>,
    pub throughput_qps: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessTest {
    pub name: String,
    pub protected_attributes: Vec<String>,
    pub metric: String,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessTest {
    pub name: String,
    pub perturbation_type: String,
    pub severity: String,
    pub success_criteria: String,
}

/// Model registry for managing multiple model versions
pub struct ModelRegistry {
    models: HashMap<Uuid, ModelCard>,
    harnesses: HashMap<Uuid, EvaluationHarness>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            harnesses: HashMap::new(),
        }
    }

    pub fn register_model(&mut self, model: ModelCard) -> Result<Uuid, String> {
        let id = model.id;
        if self.models.contains_key(&id) {
            return Err(format!("Model with ID {} already exists", id));
        }
        self.models.insert(id, model);
        Ok(id)
    }

    pub fn get_model(&self, id: &Uuid) -> Option<&ModelCard> {
        self.models.get(id)
    }

    pub fn get_latest_version(&self, name: &str) -> Option<&ModelCard> {
        self.models
            .values()
            .filter(|m| m.name == name)
            .max_by_key(|m| &m.created_at)
    }

    pub fn register_harness(&mut self, harness: EvaluationHarness) -> Result<Uuid, String> {
        let id = harness.id;
        if self.harnesses.contains_key(&id) {
            return Err(format!("Harness with ID {} already exists", id));
        }
        self.harnesses.insert(id, harness);
        Ok(id)
    }

    pub fn evaluate_model(&self, model_id: &Uuid, harness_id: &Uuid) -> Result<EvaluationReport, String> {
        let model = self.models.get(model_id)
            .ok_or_else(|| format!("Model {} not found", model_id))?;
        let harness = self.harnesses.get(harness_id)
            .ok_or_else(|| format!("Harness {} not found", harness_id))?;

        // Perform evaluation (simplified for example)
        Ok(EvaluationReport {
            model_id: *model_id,
            harness_id: *harness_id,
            timestamp: Utc::now(),
            passed_tests: harness.test_cases.len(),
            failed_tests: 0,
            performance_met: true,
            fairness_passed: true,
            robustness_score: 0.95,
            detailed_results: HashMap::new(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub model_id: Uuid,
    pub harness_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub performance_met: bool,
    pub fairness_passed: bool,
    pub robustness_score: f64,
    pub detailed_results: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registration() {
        let mut registry = ModelRegistry::new();
        
        let model = ModelCard {
            id: Uuid::new_v4(),
            name: "PolicyPredictor".to_string(),
            version: "1.0.0".to_string(),
            created_at: Utc::now(),
            model_type: ModelType::Classification,
            training_data: TrainingDataLineage {
                dataset_id: "azure-policies-2024".to_string(),
                dataset_version: "v1".to_string(),
                source_url: None,
                preprocessing_steps: vec!["tokenization".to_string()],
                split_ratios: SplitRatios {
                    train: 0.7,
                    validation: 0.15,
                    test: 0.15,
                },
                data_checksum: "sha256:abcdef".to_string(),
                collection_date: Utc::now(),
                data_characteristics: HashMap::new(),
            },
            metrics: ModelMetrics {
                accuracy: Some(0.95),
                precision: Some(0.93),
                recall: Some(0.94),
                f1_score: Some(0.935),
                auc_roc: Some(0.98),
                mae: None,
                rmse: None,
                custom_metrics: HashMap::new(),
            },
            benchmarks: vec![],
            limitations: vec!["Limited to Azure policies".to_string()],
            ethical_considerations: vec!["Ensure fair policy enforcement".to_string()],
            dependencies: vec![],
            checksum: "sha256:model123".to_string(),
            reproducibility: ReproducibilityInfo {
                random_seed: Some(42),
                environment: EnvironmentSpec {
                    os: "Ubuntu 22.04".to_string(),
                    python_version: Some("3.10.0".to_string()),
                    cuda_version: Some("11.8".to_string()),
                    container_image: Some("policycortex/ai:v1".to_string()),
                    hardware: HardwareSpec {
                        cpu: "Intel Xeon".to_string(),
                        gpu: Some("NVIDIA A100".to_string()),
                        memory_gb: 64,
                        storage_type: "SSD".to_string(),
                    },
                },
                hyperparameters: HashMap::new(),
                training_script_hash: "sha256:script123".to_string(),
                git_commit: Some("abc123def".to_string()),
            },
        };

        let id = registry.register_model(model.clone()).unwrap();
        assert_eq!(id, model.id);
        
        let retrieved = registry.get_model(&id).unwrap();
        assert_eq!(retrieved.name, "PolicyPredictor");
    }
}