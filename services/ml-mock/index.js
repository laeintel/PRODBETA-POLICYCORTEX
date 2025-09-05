import express from 'express';
import cors from 'cors';

const predictApp = express();
const explainApp = express();

predictApp.use(cors());
predictApp.use(express.json());
explainApp.use(cors());
explainApp.use(express.json());

// Mock predict endpoint
predictApp.post('/predict', (req, res) => {
  const { tenant, features } = req.body;
  res.json({
    predictions: [
      {
        resourceId: features?.resourceId || 'vm-123',
        risk: 0.72,
        confidence: 0.89,
        eta: '2h',
        topFactors: [
          { feature: 'public_exposure', weight: 0.4 },
          { feature: 'missing_encryption', weight: 0.3 },
          { feature: 'stale_credentials', weight: 0.2 }
        ]
      }
    ]
  });
});

// Mock explain endpoint
explainApp.post('/explain', (req, res) => {
  res.json({
    explanation: 'High risk due to public exposure and missing encryption',
    shap_values: [0.4, 0.3, 0.2, 0.1]
  });
});

const predictPort = process.env.ML_PREDICT_PORT || 8001;
const explainPort = process.env.ML_EXPLAIN_PORT || 8002;

predictApp.listen(predictPort, () => {
  console.log(`ML Predict mock listening on port ${predictPort}`);
});

explainApp.listen(explainPort, () => {
  console.log(`ML Explain mock listening on port ${explainPort}`);
});