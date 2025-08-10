use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Secret detection and redaction system
#[derive(Debug, Clone)]
pub struct SecretGuard {
    patterns: Vec<SecretPattern>,
    redaction_enabled: bool,
    audit_violations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretPattern {
    pub name: String,
    pub pattern: String,
    pub entropy_threshold: Option<f64>,
    pub severity: Severity,
    pub redaction_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretViolation {
    pub pattern_name: String,
    pub severity: Severity,
    pub location: String,
    pub line_number: Option<usize>,
    pub column: Option<usize>,
    pub context: String,
    pub suggested_action: String,
}

lazy_static! {
    /// Common secret patterns
    static ref SECRET_PATTERNS: Vec<SecretPattern> = vec![
        SecretPattern {
            name: "Azure Client Secret".to_string(),
            pattern: r"[a-zA-Z0-9~._-]{34,}".to_string(),
            entropy_threshold: Some(4.5),
            severity: Severity::Critical,
            redaction_text: "[AZURE_SECRET_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "AWS Access Key".to_string(),
            pattern: r"AKIA[0-9A-Z]{16}".to_string(),
            entropy_threshold: None,
            severity: Severity::Critical,
            redaction_text: "[AWS_KEY_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "AWS Secret Key".to_string(),
            pattern: r"[a-zA-Z0-9/+=]{40}".to_string(),
            entropy_threshold: Some(4.5),
            severity: Severity::Critical,
            redaction_text: "[AWS_SECRET_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "GitHub Token".to_string(),
            pattern: r"gh[ps]_[a-zA-Z0-9]{36}".to_string(),
            entropy_threshold: None,
            severity: Severity::Critical,
            redaction_text: "[GITHUB_TOKEN_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "Private Key".to_string(),
            pattern: r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----".to_string(),
            entropy_threshold: None,
            severity: Severity::Critical,
            redaction_text: "[PRIVATE_KEY_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "JWT Token".to_string(),
            pattern: r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+".to_string(),
            entropy_threshold: None,
            severity: Severity::High,
            redaction_text: "[JWT_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "Database Connection String".to_string(),
            pattern: r"(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@[^/]+/\w+".to_string(),
            entropy_threshold: None,
            severity: Severity::Critical,
            redaction_text: "[DB_CONNECTION_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "API Key".to_string(),
            // Use raw string with custom delimiter to avoid escaping quotes
            pattern: r#"api[_-]?key[_-]?[=:]\s*['\"]?[a-zA-Z0-9]{32,}['\"]?"#
                .to_string(),
            entropy_threshold: None,
            severity: Severity::High,
            redaction_text: "[API_KEY_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "Bearer Token".to_string(),
            pattern: r"[Bb]earer\s+[a-zA-Z0-9\-._~+/]+=*".to_string(),
            entropy_threshold: None,
            severity: Severity::High,
            redaction_text: "[BEARER_TOKEN_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "Credit Card".to_string(),
            pattern: r"\b(?:\d[ -]*?){13,16}\b".to_string(),
            entropy_threshold: None,
            severity: Severity::Critical,
            redaction_text: "[CREDIT_CARD_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "SSN".to_string(),
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            entropy_threshold: None,
            severity: Severity::Critical,
            redaction_text: "[SSN_REDACTED]".to_string(),
        },
        SecretPattern {
            name: "Email".to_string(),
            pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
            entropy_threshold: None,
            severity: Severity::Low,
            redaction_text: "[EMAIL_REDACTED]".to_string(),
        },
    ];
}

impl SecretGuard {
    pub fn new() -> Self {
        SecretGuard {
            patterns: SECRET_PATTERNS.clone(),
            redaction_enabled: true,
            audit_violations: true,
        }
    }

    pub fn with_custom_patterns(mut self, patterns: Vec<SecretPattern>) -> Self {
        self.patterns.extend(patterns);
        self
    }

    /// Scan text for secrets and return violations
    pub fn scan(&self, text: &str, context: &str) -> Vec<SecretViolation> {
        let mut violations = Vec::new();

        for pattern in &self.patterns {
            if let Ok(regex) = Regex::new(&pattern.pattern) {
                for mat in regex.find_iter(text) {
                    // Check entropy if threshold is set
                    if let Some(threshold) = pattern.entropy_threshold {
                        if !self.exceeds_entropy_threshold(mat.as_str(), threshold) {
                            continue;
                        }
                    }

                    violations.push(SecretViolation {
                        pattern_name: pattern.name.clone(),
                        severity: pattern.severity.clone(),
                        location: context.to_string(),
                        line_number: None,
                        column: Some(mat.start()),
                        context: self.get_context_snippet(text, mat.start(), mat.end()),
                        suggested_action: format!("Remove {} and use secure storage", pattern.name),
                    });
                }
            }
        }

        violations
    }

    /// Redact secrets in text
    pub fn redact(&self, text: &str) -> String {
        if !self.redaction_enabled {
            return text.to_string();
        }

        let mut redacted = text.to_string();

        for pattern in &self.patterns {
            if let Ok(regex) = Regex::new(&pattern.pattern) {
                redacted = regex
                    .replace_all(&redacted, pattern.redaction_text.as_str())
                    .to_string();
            }
        }

        redacted
    }

    /// Redact secrets in structured logs
    pub fn redact_json(&self, json: &serde_json::Value) -> serde_json::Value {
        match json {
            serde_json::Value::String(s) => serde_json::Value::String(self.redact(s)),
            serde_json::Value::Object(map) => {
                let mut redacted_map = serde_json::Map::new();
                for (key, value) in map {
                    // Redact common secret field names
                    if self.is_secret_field(key) {
                        redacted_map.insert(
                            key.clone(),
                            serde_json::Value::String("[REDACTED]".to_string()),
                        );
                    } else {
                        redacted_map.insert(key.clone(), self.redact_json(value));
                    }
                }
                serde_json::Value::Object(redacted_map)
            }
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| self.redact_json(v)).collect())
            }
            _ => json.clone(),
        }
    }

    fn is_secret_field(&self, field_name: &str) -> bool {
        let secret_fields = vec![
            "password",
            "secret",
            "key",
            "token",
            "api_key",
            "apikey",
            "client_secret",
            "private_key",
            "access_token",
            "refresh_token",
            "authorization",
            "auth",
            "credential",
            "cert",
            "certificate",
        ];

        let lower = field_name.to_lowercase();
        secret_fields.iter().any(|&field| lower.contains(field))
    }

    fn exceeds_entropy_threshold(&self, text: &str, threshold: f64) -> bool {
        self.calculate_shannon_entropy(text) >= threshold
    }

    fn calculate_shannon_entropy(&self, text: &str) -> f64 {
        let mut char_counts = HashMap::new();
        let len = text.len() as f64;

        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        let mut entropy = 0.0;
        for count in char_counts.values() {
            let probability = *count as f64 / len;
            entropy -= probability * probability.log2();
        }

        entropy
    }

    fn get_context_snippet(&self, text: &str, start: usize, end: usize) -> String {
        let context_size = 20;
        let snippet_start = start.saturating_sub(context_size);
        let snippet_end = (end + context_size).min(text.len());

        format!(
            "...{}...",
            &text[snippet_start..snippet_end]
                .replace('\n', " ")
                .replace('\r', "")
        )
    }
}

/// Middleware for HTTP request/response redaction
pub struct SecretRedactionMiddleware {
    guard: SecretGuard,
}

impl SecretRedactionMiddleware {
    pub fn new() -> Self {
        SecretRedactionMiddleware {
            guard: SecretGuard::new(),
        }
    }

    pub fn redact_headers(&self, headers: &mut axum::http::HeaderMap) {
        for (_name, value) in headers.iter_mut() {
            if let Ok(value_str) = value.to_str() {
                let redacted = self.guard.redact(value_str);
                if redacted != value_str {
                    *value = axum::http::HeaderValue::from_str(&redacted).unwrap_or_else(|_| {
                        axum::http::HeaderValue::from_static("[INVALID_REDACTED]")
                    });
                }
            }
        }
    }

    pub fn redact_body(&self, body: &str) -> String {
        self.guard.redact(body)
    }
}

/// Static analysis for build-time secret detection
pub struct StaticSecretAnalyzer {
    guard: SecretGuard,
    exclude_paths: Vec<String>,
}

impl StaticSecretAnalyzer {
    pub fn new() -> Self {
        StaticSecretAnalyzer {
            guard: SecretGuard::new(),
            exclude_paths: vec![
                "target/".to_string(),
                "node_modules/".to_string(),
                ".git/".to_string(),
                "dist/".to_string(),
                "build/".to_string(),
            ],
        }
    }

    pub async fn scan_directory(&self, path: &std::path::Path) -> Vec<SecretViolation> {
        let mut all_violations = Vec::new();

        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();

                // Skip excluded paths
                if self
                    .exclude_paths
                    .iter()
                    .any(|ex| path.to_string_lossy().contains(ex))
                {
                    continue;
                }

                if path.is_file() {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        let violations = self.guard.scan(&content, &path.to_string_lossy());
                        all_violations.extend(violations);
                    }
                } else if path.is_dir() {
                    let sub_violations = Box::pin(self.scan_directory(&path)).await;
                    all_violations.extend(sub_violations);
                }
            }
        }

        all_violations
    }

    pub fn generate_report(&self, violations: &[SecretViolation]) -> String {
        let mut report = String::from("Secret Detection Report\n");
        report.push_str("========================\n\n");

        if violations.is_empty() {
            report.push_str("✅ No secrets detected\n");
        } else {
            report.push_str(&format!(
                "⚠️ Found {} potential secrets\n\n",
                violations.len()
            ));

            for (i, violation) in violations.iter().enumerate() {
                report.push_str(&format!(
                    "{}. [{}] {} in {}\n   Context: {}\n   Action: {}\n\n",
                    i + 1,
                    match violation.severity {
                        Severity::Critical => "CRITICAL",
                        Severity::High => "HIGH",
                        Severity::Medium => "MEDIUM",
                        Severity::Low => "LOW",
                    },
                    violation.pattern_name,
                    violation.location,
                    violation.context,
                    violation.suggested_action
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_detection() {
        let guard = SecretGuard::new();

        let text = "My AWS key is AKIAIOSFODNN7EXAMPLE and my token is ghp_1234567890abcdef1234567890abcdef123456";
        let violations = guard.scan(text, "test.txt");

        assert!(violations.len() >= 2);
        assert!(violations.iter().any(|v| v.pattern_name.contains("AWS")));
        assert!(violations.iter().any(|v| v.pattern_name.contains("GitHub")));
    }

    #[test]
    fn test_redaction() {
        let guard = SecretGuard::new();

        let text = "Connection: postgresql://user:password123@localhost/db";
        let redacted = guard.redact(text);

        assert!(redacted.contains("[DB_CONNECTION_REDACTED]"));
        assert!(!redacted.contains("password123"));
    }

    #[test]
    fn test_json_redaction() {
        let guard = SecretGuard::new();

        let json = serde_json::json!({
            "username": "user",
            "password": "secret123",
            "api_key": "sk_test_1234567890",
            "data": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
            }
        });

        let redacted = guard.redact_json(&json);

        assert_eq!(redacted["password"], "[REDACTED]");
        assert_eq!(redacted["api_key"], "[REDACTED]");
        assert!(redacted["data"]["token"]
            .as_str()
            .unwrap()
            .contains("REDACTED"));
    }

    #[test]
    fn test_entropy_calculation() {
        let guard = SecretGuard::new();

        // High entropy (random)
        let high_entropy = "a8B3x9Z2m5K7p1Q4";
        assert!(guard.calculate_shannon_entropy(high_entropy) > 3.5);

        // Low entropy (repetitive)
        let low_entropy = "aaaaaaaaaaaaaaaa";
        assert!(guard.calculate_shannon_entropy(low_entropy) < 1.0);
    }
}
