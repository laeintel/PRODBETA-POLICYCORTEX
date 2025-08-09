use serde::{Deserialize, Serialize};
use std::env;

/// Data mode configuration for truthful data handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataMode {
    /// Real data from actual cloud providers
    Real,
    /// Simulated data for development/testing
    Simulated,
}

impl DataMode {
    /// Get current data mode from environment
    pub fn from_env() -> Self {
        match env::var("USE_REAL_DATA").as_deref() {
            Ok("true") | Ok("1") => DataMode::Real,
            _ => DataMode::Simulated,
        }
    }

    /// Check if we should use real data
    pub fn is_real(&self) -> bool {
        matches!(self, DataMode::Real)
    }

    /// Check if we should use simulated data
    pub fn is_simulated(&self) -> bool {
        matches!(self, DataMode::Simulated)
    }

    /// Get display string for UI
    pub fn display_string(&self) -> &'static str {
        match self {
            DataMode::Real => "REAL DATA",
            DataMode::Simulated => "SIMULATED",
        }
    }

    /// Get CSS class for UI styling
    pub fn css_class(&self) -> &'static str {
        match self {
            DataMode::Real => "data-mode-real",
            DataMode::Simulated => "data-mode-simulated",
        }
    }

    /// Check if operation is allowed in current mode
    pub fn allow_write_operations(&self) -> bool {
        match self {
            DataMode::Real => true,
            DataMode::Simulated => false, // Block writes in simulated mode
        }
    }
}

/// Data source indicator for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceInfo {
    pub mode: DataMode,
    pub source: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub warning: Option<String>,
}

impl DataSourceInfo {
    pub fn new(mode: DataMode) -> Self {
        let (source, warning) = match mode {
            DataMode::Real => (
                "Live Azure/AWS/GCP API".to_string(),
                None,
            ),
            DataMode::Simulated => (
                "Simulated Data".to_string(),
                Some("This is simulated data for development. Not connected to real cloud resources.".to_string()),
            ),
        };

        Self {
            mode,
            source,
            timestamp: chrono::Utc::now(),
            warning,
        }
    }
}

/// Wrapper for API responses with data source info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataResponse<T> {
    pub data: T,
    pub source_info: DataSourceInfo,
}

impl<T> DataResponse<T> {
    pub fn new(data: T, mode: DataMode) -> Self {
        Self {
            data,
            source_info: DataSourceInfo::new(mode),
        }
    }
}

/// Guard to ensure operations are allowed in current mode
pub struct DataModeGuard {
    mode: DataMode,
}

impl DataModeGuard {
    pub fn new() -> Self {
        Self {
            mode: DataMode::from_env(),
        }
    }

    /// Check if write operations are allowed
    pub fn ensure_write_allowed(&self) -> Result<(), DataModeError> {
        if !self.mode.allow_write_operations() {
            return Err(DataModeError::WriteNotAllowedInSimulatedMode);
        }
        Ok(())
    }

    /// Check if real data is required
    pub fn ensure_real_data(&self) -> Result<(), DataModeError> {
        if !self.mode.is_real() {
            return Err(DataModeError::RealDataRequired);
        }
        Ok(())
    }

    pub fn get_mode(&self) -> DataMode {
        self.mode
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DataModeError {
    #[error("Write operations are not allowed in simulated mode")]
    WriteNotAllowedInSimulatedMode,
    
    #[error("This operation requires real data mode")]
    RealDataRequired,
    
    #[error("Failed to connect to cloud provider: {0}")]
    ConnectionFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_mode_from_env() {
        env::set_var("USE_REAL_DATA", "true");
        assert_eq!(DataMode::from_env(), DataMode::Real);
        
        env::set_var("USE_REAL_DATA", "false");
        assert_eq!(DataMode::from_env(), DataMode::Simulated);
        
        env::remove_var("USE_REAL_DATA");
        assert_eq!(DataMode::from_env(), DataMode::Simulated);
    }

    #[test]
    fn test_write_operations_blocked_in_simulated() {
        let guard = DataModeGuard { mode: DataMode::Simulated };
        assert!(guard.ensure_write_allowed().is_err());
        
        let guard = DataModeGuard { mode: DataMode::Real };
        assert!(guard.ensure_write_allowed().is_ok());
    }
}