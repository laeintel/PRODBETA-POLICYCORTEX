export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Handle different edge function routes
    switch (url.pathname) {
      case '/edge/compliance-check':
        return handleComplianceCheck(request);
      case '/edge/policy-evaluate':
        return handlePolicyEvaluation(request);
      case '/edge/anomaly-detect':
        return handleAnomalyDetection(request);
      case '/edge/health':
        return new Response(JSON.stringify({ status: 'healthy', edge: true }), {
          headers: { 'Content-Type': 'application/json' },
        });
      default:
        return new Response('Edge function not found', { status: 404 });
    }
  },
};

async function handleComplianceCheck(request) {
  try {
    const body = await request.json();
    
    // Simulated compliance check logic
    const complianceScore = Math.random() * 100;
    const isCompliant = complianceScore > 70;
    
    return new Response(JSON.stringify({
      resource: body.resource,
      policy: body.policy,
      compliant: isCompliant,
      score: complianceScore.toFixed(2),
      timestamp: new Date().toISOString(),
      processedAt: 'edge',
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Invalid request' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

async function handlePolicyEvaluation(request) {
  try {
    const body = await request.json();
    
    // Simulated policy evaluation
    const violations = [];
    if (Math.random() > 0.5) {
      violations.push({
        rule: 'encryption-required',
        severity: 'high',
        message: 'Resource must have encryption enabled',
      });
    }
    
    return new Response(JSON.stringify({
      resource: body.resource,
      violations,
      evaluated: true,
      timestamp: new Date().toISOString(),
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Invalid request' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

async function handleAnomalyDetection(request) {
  try {
    const body = await request.json();
    
    // Simulated anomaly detection
    const anomalyScore = Math.random();
    const isAnomaly = anomalyScore > 0.8;
    
    return new Response(JSON.stringify({
      data: body.data,
      anomaly: isAnomaly,
      confidence: (1 - anomalyScore).toFixed(3),
      timestamp: new Date().toISOString(),
      model: 'edge-anomaly-detector-v1',
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Invalid request' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}