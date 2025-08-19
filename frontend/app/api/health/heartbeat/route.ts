import { NextRequest, NextResponse } from 'next/server';

interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency?: number;
  lastCheck: string;
}

interface HeartbeatResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: ServiceHealth[];
  metrics: {
    uptime: number;
    requestsPerMinute: number;
    averageLatency: number;
    errorRate: number;
  };
  version: string;
  environment: string;
}

// Simple in-memory metrics (in production, use proper metrics store)
let startTime = Date.now();
let requestCount = 0;
let totalLatency = 0;
let errorCount = 0;

async function checkService(url: string, name: string): Promise<ServiceHealth> {
  const start = Date.now();
  try {
    const response = await fetch(url, { 
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });
    const latency = Date.now() - start;
    
    return {
      name,
      status: response.ok ? 'healthy' : 'degraded',
      latency,
      lastCheck: new Date().toISOString()
    };
  } catch (error) {
    return {
      name,
      status: 'unhealthy',
      latency: Date.now() - start,
      lastCheck: new Date().toISOString()
    };
  }
}

export async function GET(request: NextRequest) {
  requestCount++;
  const requestStart = Date.now();

  try {
    // Check dependent services
    const services = await Promise.all([
      checkService('http://localhost:8080/health', 'Core API'),
      // Apollo Server exposes a standard health endpoint here
      checkService('http://localhost:4000/.well-known/apollo/server-health', 'GraphQL Gateway'),
      checkService('http://localhost:5432', 'PostgreSQL'),
      checkService('http://localhost:6379', 'Redis Cache')
    ]);

    // Calculate overall status
    const unhealthyCount = services.filter(s => s.status === 'unhealthy').length;
    const degradedCount = services.filter(s => s.status === 'degraded').length;
    
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (unhealthyCount > 0) {
      overallStatus = 'unhealthy';
    } else if (degradedCount > 0) {
      overallStatus = 'degraded';
    }

    // Calculate metrics
    const uptime = Math.floor((Date.now() - startTime) / 1000); // in seconds
    const requestsPerMinute = (requestCount / (uptime / 60)) || 0;
    const averageLatency = totalLatency / requestCount || 0;
    const errorRate = (errorCount / requestCount) * 100 || 0;

    const response: HeartbeatResponse = {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      services,
      metrics: {
        uptime,
        requestsPerMinute: Math.round(requestsPerMinute),
        averageLatency: Math.round(averageLatency),
        errorRate: Math.round(errorRate * 100) / 100
      },
      version: process.env.npm_package_version || '2.0.0',
      environment: process.env.NODE_ENV || 'development'
    };

    // Update metrics
    totalLatency += (Date.now() - requestStart);

    // Export metrics in Prometheus format if requested
    const acceptHeader = request.headers.get('accept');
    if (acceptHeader && acceptHeader.includes('text/plain')) {
      const prometheusMetrics = `
# HELP policycortex_up Whether PolicyCortex is up (1) or down (0)
# TYPE policycortex_up gauge
policycortex_up ${overallStatus === 'healthy' ? 1 : 0}

# HELP policycortex_uptime_seconds Uptime in seconds
# TYPE policycortex_uptime_seconds counter
policycortex_uptime_seconds ${uptime}

# HELP policycortex_requests_per_minute Requests per minute
# TYPE policycortex_requests_per_minute gauge
policycortex_requests_per_minute ${requestsPerMinute}

# HELP policycortex_average_latency_ms Average latency in milliseconds
# TYPE policycortex_average_latency_ms gauge
policycortex_average_latency_ms ${averageLatency}

# HELP policycortex_error_rate Error rate percentage
# TYPE policycortex_error_rate gauge
policycortex_error_rate ${errorRate}

# HELP policycortex_service_health Service health status (1=healthy, 0.5=degraded, 0=unhealthy)
# TYPE policycortex_service_health gauge
${services.map(s => `policycortex_service_health{service="${s.name}"} ${s.status === 'healthy' ? 1 : s.status === 'degraded' ? 0.5 : 0}`).join('\n')}

# HELP policycortex_service_latency_ms Service latency in milliseconds
# TYPE policycortex_service_latency_ms gauge
${services.map(s => `policycortex_service_latency_ms{service="${s.name}"} ${s.latency || 0}`).join('\n')}
`.trim();

      return new NextResponse(prometheusMetrics, {
        status: 200,
        headers: {
          'Content-Type': 'text/plain; version=0.0.4',
          'Cache-Control': 'no-cache, no-store, must-revalidate'
        }
      });
    }

    return NextResponse.json(response, {
      status: overallStatus === 'healthy' ? 200 : overallStatus === 'degraded' ? 203 : 503,
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'X-Health-Status': overallStatus
      }
    });

  } catch (error) {
    errorCount++;
    totalLatency += (Date.now() - requestStart);
    
    return NextResponse.json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Failed to check service health',
      services: [],
      metrics: {
        uptime: Math.floor((Date.now() - startTime) / 1000),
        requestsPerMinute: 0,
        averageLatency: 0,
        errorRate: 100
      },
      version: process.env.npm_package_version || '2.0.0',
      environment: process.env.NODE_ENV || 'development'
    }, { 
      status: 503,
      headers: {
        'X-Health-Status': 'unhealthy'
      }
    });
  }
}