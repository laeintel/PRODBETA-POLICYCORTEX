import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  IconButton,
  Chip,
  Stack,
  LinearProgress,
  Alert,
  AlertTitle,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  Skeleton,
  Tooltip,
  Badge,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Slider,
  Switch,
  FormGroup,
  FormControlLabel,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Analytics,
  AutoGraph,
  PsychologyAlt,
  BubbleChart,
  DonutLarge,
  Timeline,
  Insights,
  Speed,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Download,
  Share,
  Settings,
  FilterList,
  DateRange,
  Lightbulb,
  AutoAwesome,
  Psychology,
  ModelTraining,
  BarChart,
  ShowChart,
  PieChart,
  ScatterPlot,
  HeatmapOutlined,
  Radar,
  TableChart,
  Functions,
  DataUsage,
  QueryStats,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart as RechartsBarChart,
  Bar,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar as RadarArea,
  ScatterChart,
  Scatter,
  ZAxis,
  Treemap,
  ComposedChart,
  ReferenceLine,
  ReferenceArea,
  Brush,
} from 'recharts';
import { format, subDays, startOfMonth, endOfMonth } from 'date-fns';
import { api } from '../../services/api';

interface AIInsight {
  id: string;
  type: 'anomaly' | 'trend' | 'prediction' | 'recommendation' | 'optimization';
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  impact: string;
  confidence: number;
  actions: string[];
  data?: any;
  timestamp: string;
}

interface PredictiveModel {
  id: string;
  name: string;
  type: string;
  accuracy: number;
  lastTrained: string;
  predictions: any[];
  confidence: number;
  status: 'active' | 'training' | 'inactive';
}

interface AnalyticsMetric {
  name: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  forecast: number[];
  anomalies: any[];
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0', '#ffb347', '#67b7dc'];

export const AIAnalyticsDashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');
  const [viewMode, setViewMode] = useState<'insights' | 'predictive' | 'correlation' | 'optimization'>('insights');
  const [selectedMetric, setSelectedMetric] = useState('all');
  const [aiInsights, setAiInsights] = useState<AIInsight[]>([]);
  const [predictiveModels, setPredictiveModels] = useState<PredictiveModel[]>([]);
  const [analyticsData, setAnalyticsData] = useState<any>({});
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [selectedInsight, setSelectedInsight] = useState<AIInsight | null>(null);
  const [modelTrainingOpen, setModelTrainingOpen] = useState(false);
  const [correlationMatrix, setCorrelationMatrix] = useState<any[]>([]);
  const [optimizationSuggestions, setOptimizationSuggestions] = useState<any[]>([]);

  useEffect(() => {
    loadAnalyticsData();
    if (autoRefresh) {
      const interval = setInterval(loadAnalyticsData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [timeRange, selectedMetric, autoRefresh]);

  const loadAnalyticsData = async () => {
    setLoading(true);
    try {
      // Load AI insights
      const insightsResponse = await api.get('/api/v1/analytics/ai-insights', {
        params: { timeRange, metric: selectedMetric },
      });
      setAiInsights(insightsResponse.data);

      // Load predictive models
      const modelsResponse = await api.get('/api/v1/analytics/predictive-models');
      setPredictiveModels(modelsResponse.data);

      // Load analytics data based on view mode
      const analyticsResponse = await api.get(`/api/v1/analytics/${viewMode}`, {
        params: { timeRange },
      });
      setAnalyticsData(analyticsResponse.data);

      // Load correlation matrix
      if (viewMode === 'correlation') {
        const correlationResponse = await api.get('/api/v1/analytics/correlation-matrix');
        setCorrelationMatrix(correlationResponse.data);
      }

      // Load optimization suggestions
      if (viewMode === 'optimization') {
        const optimizationResponse = await api.get('/api/v1/analytics/optimization-suggestions');
        setOptimizationSuggestions(optimizationResponse.data);
      }
    } catch (error) {
      console.error('Failed to load analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async (modelId: string) => {
    try {
      await api.post(`/api/v1/analytics/models/${modelId}/train`);
      loadAnalyticsData();
    } catch (error) {
      console.error('Failed to train model:', error);
    }
  };

  const applyOptimization = async (optimizationId: string) => {
    try {
      await api.post(`/api/v1/analytics/optimizations/${optimizationId}/apply`);
      loadAnalyticsData();
    } catch (error) {
      console.error('Failed to apply optimization:', error);
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'anomaly':
        return <BubbleChart color="error" />;
      case 'trend':
        return <Timeline color="info" />;
      case 'prediction':
        return <AutoGraph color="primary" />;
      case 'recommendation':
        return <Lightbulb color="warning" />;
      case 'optimization':
        return <AutoAwesome color="success" />;
      default:
        return <Insights />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  // Mock data generation for demo
  const generateMockData = () => {
    const data = [];
    for (let i = 0; i < 30; i++) {
      data.push({
        date: format(subDays(new Date(), 30 - i), 'MMM dd'),
        actual: Math.floor(Math.random() * 100) + 50,
        predicted: Math.floor(Math.random() * 100) + 45,
        anomaly: Math.random() > 0.9 ? Math.floor(Math.random() * 50) + 100 : null,
        confidence: Math.random() * 0.3 + 0.7,
      });
    }
    return data;
  };

  const timeSeriesData = generateMockData();

  const correlationHeatmapData = [
    { metric1: 'Cost', metric2: 'Resources', correlation: 0.85 },
    { metric1: 'Cost', metric2: 'Compliance', correlation: -0.3 },
    { metric1: 'Cost', metric2: 'Performance', correlation: 0.6 },
    { metric1: 'Resources', metric2: 'Compliance', correlation: 0.4 },
    { metric1: 'Resources', metric2: 'Performance', correlation: 0.75 },
    { metric1: 'Compliance', metric2: 'Performance', correlation: -0.2 },
  ];

  const radarData = [
    { metric: 'Security', current: 85, optimal: 95, predicted: 88 },
    { metric: 'Compliance', current: 78, optimal: 90, predicted: 82 },
    { metric: 'Cost', current: 65, optimal: 80, predicted: 70 },
    { metric: 'Performance', current: 92, optimal: 95, predicted: 93 },
    { metric: 'Reliability', current: 88, optimal: 95, predicted: 90 },
    { metric: 'Scalability', current: 75, optimal: 85, predicted: 78 },
  ];

  const renderInsightsView = () => (
    <Grid container spacing={3}>
      {/* AI Insights Summary */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
              <PsychologyAlt sx={{ mr: 1 }} />
              AI-Powered Insights
            </Typography>
            <Stack direction="row" spacing={1}>
              <Chip
                label={`${aiInsights.filter(i => i.severity === 'critical').length} Critical`}
                color="error"
                size="small"
              />
              <Chip
                label={`${aiInsights.filter(i => i.severity === 'high').length} High`}
                color="warning"
                size="small"
              />
              <Chip
                label={`${aiInsights.filter(i => i.confidence > 0.8).length} High Confidence`}
                color="success"
                size="small"
              />
            </Stack>
          </Box>

          <Grid container spacing={2}>
            {aiInsights.slice(0, 6).map((insight) => (
              <Grid item xs={12} md={6} lg={4} key={insight.id}>
                <Card
                  sx={{
                    height: '100%',
                    cursor: 'pointer',
                    '&:hover': { boxShadow: 3 },
                  }}
                  onClick={() => setSelectedInsight(insight)}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      {getInsightIcon(insight.type)}
                      <Chip
                        label={insight.severity}
                        color={getSeverityColor(insight.severity)}
                        size="small"
                      />
                    </Box>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                      {insight.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {insight.description}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <LinearProgress
                        variant="determinate"
                        value={insight.confidence * 100}
                        sx={{ flexGrow: 1, mr: 1, height: 6, borderRadius: 3 }}
                      />
                      <Typography variant="caption">
                        {(insight.confidence * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </CardContent>
                  <CardActions>
                    <Button size="small">View Details</Button>
                    <Button size="small" color="primary">
                      Take Action
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Time Series with Anomaly Detection */}
      <Grid item xs={12} lg={8}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Predictive Analytics & Anomaly Detection
          </Typography>
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <RechartsTooltip />
              <Legend />
              <Area
                yAxisId="left"
                type="monotone"
                dataKey="predicted"
                fill="#8884d8"
                stroke="#8884d8"
                fillOpacity={0.3}
                name="Predicted"
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="actual"
                stroke="#82ca9d"
                strokeWidth={2}
                name="Actual"
              />
              <Scatter
                yAxisId="left"
                dataKey="anomaly"
                fill="#ff0000"
                name="Anomaly"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="confidence"
                stroke="#ffc658"
                strokeDasharray="5 5"
                name="Confidence"
              />
              <Brush dataKey="date" height={30} stroke="#8884d8" />
            </ComposedChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Model Performance */}
      <Grid item xs={12} lg={4}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            ML Model Performance
          </Typography>
          <List>
            {predictiveModels.map((model) => (
              <ListItem key={model.id} divider>
                <ListItemIcon>
                  <Badge
                    badgeContent={
                      model.status === 'active' ? (
                        <CheckCircle color="success" fontSize="small" />
                      ) : model.status === 'training' ? (
                        <CircularProgress size={12} />
                      ) : (
                        <Error color="error" fontSize="small" />
                      )
                    }
                  >
                    <ModelTraining />
                  </Badge>
                </ListItemIcon>
                <ListItemText
                  primary={model.name}
                  secondary={
                    <Box>
                      <Typography variant="caption" display="block">
                        Accuracy: {(model.accuracy * 100).toFixed(1)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={model.accuracy * 100}
                        sx={{ mt: 0.5, height: 4, borderRadius: 2 }}
                      />
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    onClick={() => trainModel(model.id)}
                    disabled={model.status === 'training'}
                  >
                    <Refresh />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<Settings />}
            sx={{ mt: 2 }}
            onClick={() => setModelTrainingOpen(true)}
          >
            Configure Models
          </Button>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderPredictiveView = () => (
    <Grid container spacing={3}>
      {/* Multi-Metric Predictions */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Multi-Metric Predictive Analysis
          </Typography>
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <RadarArea
                name="Current"
                dataKey="current"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.6}
              />
              <RadarArea
                name="Predicted"
                dataKey="predicted"
                stroke="#82ca9d"
                fill="#82ca9d"
                fillOpacity={0.6}
              />
              <RadarArea
                name="Optimal"
                dataKey="optimal"
                stroke="#ffc658"
                fill="#ffc658"
                fillOpacity={0.3}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Forecast Cards */}
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {['Cost', 'Performance', 'Compliance', 'Security'].map((metric) => (
            <Grid item xs={12} sm={6} md={3} key={metric}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">
                    {metric} Forecast (30 days)
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'baseline', mt: 1 }}>
                    <Typography variant="h4">
                      {Math.floor(Math.random() * 20) + 70}%
                    </Typography>
                    <Box sx={{ ml: 2 }}>
                      {Math.random() > 0.5 ? (
                        <TrendingUp color="success" />
                      ) : (
                        <TrendingDown color="error" />
                      )}
                    </Box>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Confidence: {(Math.random() * 0.2 + 0.8).toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Grid>
    </Grid>
  );

  const renderCorrelationView = () => (
    <Grid container spacing={3}>
      {/* Correlation Heatmap */}
      <Grid item xs={12} lg={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Cross-Domain Correlation Analysis
          </Typography>
          <Box sx={{ mt: 2 }}>
            {['Cost', 'Resources', 'Compliance', 'Performance'].map((metric1, i) => (
              <Box key={metric1} sx={{ display: 'flex', mb: 1 }}>
                <Typography sx={{ width: 100, fontSize: 12 }}>{metric1}</Typography>
                {['Cost', 'Resources', 'Compliance', 'Performance'].map((metric2, j) => {
                  const correlation = i === j ? 1 : Math.random() * 2 - 1;
                  const color = correlation > 0.5 
                    ? `rgba(76, 175, 80, ${correlation})` 
                    : correlation < -0.5 
                    ? `rgba(244, 67, 54, ${Math.abs(correlation)})`
                    : `rgba(158, 158, 158, ${Math.abs(correlation)})`;
                  
                  return (
                    <Tooltip key={metric2} title={`${metric1} vs ${metric2}: ${correlation.toFixed(2)}`}>
                      <Box
                        sx={{
                          width: 40,
                          height: 40,
                          bgcolor: color,
                          border: '1px solid #e0e0e0',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 10,
                          color: Math.abs(correlation) > 0.5 ? 'white' : 'black',
                          cursor: 'pointer',
                        }}
                      >
                        {correlation.toFixed(1)}
                      </Box>
                    </Tooltip>
                  );
                })}
              </Box>
            ))}
          </Box>
        </Paper>
      </Grid>

      {/* Scatter Plot Correlation */}
      <Grid item xs={12} lg={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resource vs Cost Correlation
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="resources" name="Resources" unit=" units" />
              <YAxis dataKey="cost" name="Cost" unit="$" />
              <ZAxis dataKey="compliance" name="Compliance" range={[50, 400]} />
              <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter
                name="Data Points"
                data={Array.from({ length: 50 }, () => ({
                  resources: Math.floor(Math.random() * 100),
                  cost: Math.floor(Math.random() * 10000),
                  compliance: Math.floor(Math.random() * 100),
                }))}
                fill="#8884d8"
              />
              <ReferenceLine y={5000} stroke="red" strokeDasharray="3 3" label="Cost Threshold" />
              <ReferenceLine x={50} stroke="green" strokeDasharray="3 3" label="Resource Target" />
            </ScatterChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Correlation Insights */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Correlation-Based Insights
          </Typography>
          <Grid container spacing={2}>
            {[
              {
                title: 'Strong Positive Correlation',
                metrics: 'Resources ↔ Cost',
                value: 0.85,
                insight: 'Increasing resources directly impacts costs. Consider optimization.',
              },
              {
                title: 'Strong Negative Correlation',
                metrics: 'Security ↔ Performance',
                value: -0.72,
                insight: 'Security measures may impact performance. Balance required.',
              },
              {
                title: 'Unexpected Correlation',
                metrics: 'Compliance ↔ User Satisfaction',
                value: 0.63,
                insight: 'Better compliance correlates with user satisfaction. Prioritize compliance.',
              },
            ].map((item, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary">
                      {item.title}
                    </Typography>
                    <Typography variant="h6" sx={{ my: 1 }}>
                      {item.metrics}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={Math.abs(item.value) * 100}
                        sx={{
                          flexGrow: 1,
                          height: 8,
                          borderRadius: 4,
                          bgcolor: 'grey.200',
                          '& .MuiLinearProgress-bar': {
                            bgcolor: item.value > 0 ? 'success.main' : 'error.main',
                          },
                        }}
                      />
                      <Typography variant="body2" sx={{ ml: 1 }}>
                        {item.value}
                      </Typography>
                    </Box>
                    <Typography variant="caption">
                      {item.insight}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderOptimizationView = () => (
    <Grid container spacing={3}>
      {/* Optimization Opportunities */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            AI-Recommended Optimizations
          </Typography>
          <List>
            {[
              {
                id: '1',
                title: 'Resource Rightsizing',
                impact: 'High',
                savings: '$12,500/month',
                effort: 'Low',
                confidence: 0.92,
                description: 'Identified 23 over-provisioned resources that can be rightsized',
              },
              {
                id: '2',
                title: 'Automated Scaling Policy',
                impact: 'Medium',
                savings: '$8,200/month',
                effort: 'Medium',
                confidence: 0.85,
                description: 'Implement auto-scaling for 12 services based on usage patterns',
              },
              {
                id: '3',
                title: 'Reserved Instance Optimization',
                impact: 'High',
                savings: '$18,000/month',
                effort: 'Low',
                confidence: 0.95,
                description: 'Convert 45 on-demand instances to reserved instances',
              },
            ].map((optimization) => (
              <ListItem key={optimization.id} divider>
                <ListItemIcon>
                  <AutoAwesome color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle1">{optimization.title}</Typography>
                      <Chip label={optimization.impact} size="small" color="primary" />
                      <Chip label={optimization.effort} size="small" variant="outlined" />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {optimization.description}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <Typography variant="caption" sx={{ mr: 2 }}>
                          Potential Savings: <strong>{optimization.savings}</strong>
                        </Typography>
                        <Typography variant="caption">
                          Confidence: {(optimization.confidence * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Button
                    variant="contained"
                    size="small"
                    onClick={() => applyOptimization(optimization.id)}
                  >
                    Apply
                  </Button>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Paper>
      </Grid>

      {/* Optimization Impact Analysis */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Projected Impact (6 months)
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart
              data={Array.from({ length: 6 }, (_, i) => ({
                month: format(new Date(2024, i), 'MMM'),
                current: 100000 - i * 1000,
                optimized: 100000 - i * 5000 - i * i * 1000,
                savings: i * 4000 + i * i * 1000,
              }))}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey="current"
                stackId="1"
                stroke="#8884d8"
                fill="#8884d8"
                name="Current Cost"
              />
              <Area
                type="monotone"
                dataKey="optimized"
                stackId="2"
                stroke="#82ca9d"
                fill="#82ca9d"
                name="Optimized Cost"
              />
              <Area
                type="monotone"
                dataKey="savings"
                stackId="3"
                stroke="#ffc658"
                fill="#ffc658"
                name="Savings"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Optimization Metrics */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Optimization Metrics
          </Typography>
          <Grid container spacing={2}>
            {[
              { label: 'Total Savings Potential', value: '$38,700/month', color: 'success' },
              { label: 'Implementation Effort', value: '~2 weeks', color: 'info' },
              { label: 'ROI', value: '485%', color: 'primary' },
              { label: 'Risk Level', value: 'Low', color: 'success' },
            ].map((metric) => (
              <Grid item xs={6} key={metric.label}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="caption" color="text.secondary">
                      {metric.label}
                    </Typography>
                    <Typography variant="h6" color={`${metric.color}.main`}>
                      {metric.value}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} md={6} key={i}>
              <Skeleton variant="rectangular" height={300} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          AI-Powered Analytics
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, value) => value && setViewMode(value)}
            size="small"
          >
            <ToggleButton value="insights">
              <Insights sx={{ mr: 1 }} />
              Insights
            </ToggleButton>
            <ToggleButton value="predictive">
              <AutoGraph sx={{ mr: 1 }} />
              Predictive
            </ToggleButton>
            <ToggleButton value="correlation">
              <BubbleChart sx={{ mr: 1 }} />
              Correlation
            </ToggleButton>
            <ToggleButton value="optimization">
              <AutoAwesome sx={{ mr: 1 }} />
              Optimization
            </ToggleButton>
          </ToggleButtonGroup>

          <Stack direction="row" spacing={2}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                label="Time Range"
                onChange={(e) => setTimeRange(e.target.value)}
              >
                <MenuItem value="24h">24 Hours</MenuItem>
                <MenuItem value="7d">7 Days</MenuItem>
                <MenuItem value="30d">30 Days</MenuItem>
                <MenuItem value="90d">90 Days</MenuItem>
              </Select>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
              }
              label="Auto Refresh"
            />
            
            <IconButton onClick={loadAnalyticsData}>
              <Refresh />
            </IconButton>
          </Stack>
        </Box>
      </Box>

      {/* View Content */}
      {viewMode === 'insights' && renderInsightsView()}
      {viewMode === 'predictive' && renderPredictiveView()}
      {viewMode === 'correlation' && renderCorrelationView()}
      {viewMode === 'optimization' && renderOptimizationView()}

      {/* Insight Details Dialog */}
      <Dialog
        open={!!selectedInsight}
        onClose={() => setSelectedInsight(null)}
        maxWidth="md"
        fullWidth
      >
        {selectedInsight && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                {getInsightIcon(selectedInsight.type)}
                <Typography sx={{ ml: 1 }}>{selectedInsight.title}</Typography>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Stack spacing={2}>
                <Alert severity={getSeverityColor(selectedInsight.severity) as any}>
                  <AlertTitle>Impact</AlertTitle>
                  {selectedInsight.impact}
                </Alert>
                <Typography variant="body1">{selectedInsight.description}</Typography>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Recommended Actions:
                  </Typography>
                  <List dense>
                    {selectedInsight.actions.map((action, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckCircle fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={action} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="caption">
                    Confidence: {(selectedInsight.confidence * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="caption">
                    Generated: {format(new Date(selectedInsight.timestamp), 'PPp')}
                  </Typography>
                </Box>
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedInsight(null)}>Close</Button>
              <Button variant="contained" onClick={() => setSelectedInsight(null)}>
                Take Action
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* Model Training Dialog */}
      <Dialog
        open={modelTrainingOpen}
        onClose={() => setModelTrainingOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Configure ML Models</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Training Data Range"
              type="number"
              defaultValue={90}
              InputProps={{ endAdornment: 'days' }}
            />
            <FormControl fullWidth>
              <InputLabel>Model Type</InputLabel>
              <Select defaultValue="lstm">
                <MenuItem value="lstm">LSTM (Time Series)</MenuItem>
                <MenuItem value="xgboost">XGBoost (Classification)</MenuItem>
                <MenuItem value="prophet">Prophet (Forecasting)</MenuItem>
                <MenuItem value="ensemble">Ensemble</MenuItem>
              </Select>
            </FormControl>
            <TextField
              fullWidth
              label="Hyperparameters"
              multiline
              rows={3}
              defaultValue='{"learning_rate": 0.001, "epochs": 100}'
            />
            <FormGroup>
              <FormControlLabel control={<Switch defaultChecked />} label="Auto-retrain weekly" />
              <FormControlLabel control={<Switch />} label="Enable A/B testing" />
              <FormControlLabel control={<Switch defaultChecked />} label="Monitor drift" />
            </FormGroup>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModelTrainingOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setModelTrainingOpen(false)}>
            Start Training
          </Button>
        </DialogActions>
      </Dialog>

      {/* Speed Dial Actions */}
      <SpeedDial
        ariaLabel="Analytics Actions"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        icon={<SpeedDialIcon />}
      >
        <SpeedDialAction
          icon={<Download />}
          tooltipTitle="Export Report"
          onClick={() => {}}
        />
        <SpeedDialAction
          icon={<Share />}
          tooltipTitle="Share Dashboard"
          onClick={() => {}}
        />
        <SpeedDialAction
          icon={<Settings />}
          tooltipTitle="Configure"
          onClick={() => {}}
        />
      </SpeedDial>
    </Box>
  );
};