import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Button,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Alert,
  AlertTitle,
  Stack,
  Tabs,
  Tab,
  CircularProgress,
  Badge,
  Tooltip,
  Menu,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Error,
  Warning,
  Info,
  Refresh,
  Download,
  FilterList,
  Search,
  Assessment,
  Security,
  Policy,
  Schedule,
  CloudUpload,
  Rule,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../../services/api';
import { format } from 'date-fns';

interface ComplianceMetrics {
  overallScore: number;
  complianceLevel: string;
  totalResources: number;
  compliantResources: number;
  nonCompliantResources: number;
  criticalViolations: number;
  highViolations: number;
  mediumViolations: number;
  lowViolations: number;
}

interface PolicyCoverage {
  policyName: string;
  coverage: number;
}

interface TrendData {
  date: string;
  score: number;
  violations: number;
}

interface ResourceCompliance {
  resourceId: string;
  resourceName: string;
  resourceType: string;
  complianceStatus: string;
  complianceScore: number;
  violations: any[];
  lastChecked: string;
}

export const ComplianceDashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  const [metrics, setMetrics] = useState<ComplianceMetrics | null>(null);
  const [resources, setResources] = useState<ResourceCompliance[]>([]);
  const [policyCoverage, setPolicyCoverage] = useState<PolicyCoverage[]>([]);
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [filterStatus, setFilterStatus] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  useEffect(() => {
    loadComplianceData();
  }, [selectedTimeRange, filterStatus]);

  const loadComplianceData = async () => {
    setLoading(true);
    try {
      // Load compliance metrics
      const metricsResponse = await api.get('/api/v1/compliance/metrics');
      setMetrics(metricsResponse.data);

      // Load resource compliance
      const resourcesResponse = await api.get('/api/v1/compliance/resources', {
        params: { status: filterStatus !== 'all' ? filterStatus : undefined },
      });
      setResources(resourcesResponse.data);

      // Load policy coverage
      const coverageResponse = await api.get('/api/v1/compliance/coverage');
      setPolicyCoverage(coverageResponse.data);

      // Load trend data
      const trendResponse = await api.get('/api/v1/compliance/trends', {
        params: { range: selectedTimeRange },
      });
      setTrendData(trendResponse.data);
    } catch (error) {
      console.error('Failed to load compliance data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getComplianceLevelColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'excellent':
        return 'success';
      case 'good':
        return 'info';
      case 'fair':
        return 'warning';
      case 'poor':
        return 'error';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'compliant':
        return <CheckCircle color="success" />;
      case 'non_compliant':
        return <Error color="error" />;
      case 'partially_compliant':
        return <Warning color="warning" />;
      default:
        return <Info color="info" />;
    }
  };

  const handleExportReport = async () => {
    try {
      const response = await api.get('/api/v1/compliance/export', {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `compliance_report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Failed to export report:', error);
    }
  };

  const filteredResources = resources.filter((resource) =>
    resource.resourceName.toLowerCase().includes(searchQuery.toLowerCase()) ||
    resource.resourceType.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const violationPieData = metrics
    ? [
        { name: 'Critical', value: metrics.criticalViolations, color: '#f44336' },
        { name: 'High', value: metrics.highViolations, color: '#ff9800' },
        { name: 'Medium', value: metrics.mediumViolations, color: '#ffc107' },
        { name: 'Low', value: metrics.lowViolations, color: '#4caf50' },
      ]
    : [];

  const complianceGaugeData = metrics
    ? [
        {
          name: 'Compliant',
          value: metrics.compliantResources,
          color: '#4caf50',
        },
        {
          name: 'Non-Compliant',
          value: metrics.nonCompliantResources,
          color: '#f44336',
        },
      ]
    : [];

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="h4" fontWeight="bold">
          Compliance Dashboard
        </Typography>
        <Stack direction="row" spacing={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={selectedTimeRange}
              label="Time Range"
              onChange={(e) => setSelectedTimeRange(e.target.value)}
            >
              <MenuItem value="24h">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
              <MenuItem value="90d">Last 90 Days</MenuItem>
            </Select>
          </FormControl>
          <Button
            startIcon={<Refresh />}
            onClick={loadComplianceData}
            variant="outlined"
          >
            Refresh
          </Button>
          <Button
            startIcon={<Download />}
            onClick={handleExportReport}
            variant="contained"
          >
            Export Report
          </Button>
        </Stack>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="subtitle2" color="text.secondary">
                  Compliance Score
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'baseline' }}>
                  <Typography variant="h3" fontWeight="bold">
                    {metrics?.overallScore.toFixed(1)}%
                  </Typography>
                  {trendData.length > 1 && (
                    <Box sx={{ ml: 2 }}>
                      {trendData[trendData.length - 1].score >
                      trendData[trendData.length - 2].score ? (
                        <TrendingUp color="success" />
                      ) : (
                        <TrendingDown color="error" />
                      )}
                    </Box>
                  )}
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={metrics?.overallScore || 0}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    bgcolor: 'grey.200',
                    '& .MuiLinearProgress-bar': {
                      bgcolor:
                        (metrics?.overallScore || 0) > 80
                          ? 'success.main'
                          : (metrics?.overallScore || 0) > 60
                          ? 'warning.main'
                          : 'error.main',
                    },
                  }}
                />
                <Chip
                  label={metrics?.complianceLevel}
                  color={getComplianceLevelColor(metrics?.complianceLevel || '')}
                  size="small"
                />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Stack spacing={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Total Resources
                </Typography>
                <Typography variant="h4" fontWeight="bold">
                  {metrics?.totalResources || 0}
                </Typography>
                <Stack direction="row" spacing={1}>
                  <Chip
                    icon={<CheckCircle />}
                    label={`${metrics?.compliantResources || 0} Compliant`}
                    color="success"
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={<Error />}
                    label={`${metrics?.nonCompliantResources || 0} Non-Compliant`}
                    color="error"
                    size="small"
                    variant="outlined"
                  />
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Stack spacing={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Critical Violations
                </Typography>
                <Typography variant="h4" fontWeight="bold" color="error.main">
                  {metrics?.criticalViolations || 0}
                </Typography>
                <Stack direction="row" spacing={0.5}>
                  <Badge badgeContent={metrics?.highViolations} color="warning">
                    <Chip label="High" size="small" />
                  </Badge>
                  <Badge badgeContent={metrics?.mediumViolations} color="info">
                    <Chip label="Medium" size="small" />
                  </Badge>
                  <Badge badgeContent={metrics?.lowViolations} color="success">
                    <Chip label="Low" size="small" />
                  </Badge>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Stack spacing={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Policy Coverage
                </Typography>
                <Typography variant="h4" fontWeight="bold">
                  {policyCoverage.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Active Policies
                </Typography>
                <Button
                  size="small"
                  startIcon={<Policy />}
                  variant="text"
                  href="/compliance/policies"
                >
                  Manage Policies
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Compliance Trend
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <ChartTooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="score"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                  name="Compliance Score"
                />
                <Line
                  type="monotone"
                  dataKey="violations"
                  stroke="#ff7300"
                  name="Violations"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Violations by Severity
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={violationPieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {violationPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <ChartTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Policy Coverage */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Policy Coverage Analysis
        </Typography>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={policyCoverage.slice(0, 10)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="policyName" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <ChartTooltip />
            <Bar dataKey="coverage" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </Paper>

      {/* Resources Table */}
      <Paper sx={{ p: 3 }}>
        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="h6">Resource Compliance</Typography>
          <Stack direction="row" spacing={2}>
            <TextField
              size="small"
              placeholder="Search resources..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Filter Status</InputLabel>
              <Select
                value={filterStatus}
                label="Filter Status"
                onChange={(e) => setFilterStatus(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="compliant">Compliant</MenuItem>
                <MenuItem value="non_compliant">Non-Compliant</MenuItem>
                <MenuItem value="partially_compliant">Partially Compliant</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </Box>

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Status</TableCell>
                <TableCell>Resource Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Compliance Score</TableCell>
                <TableCell>Violations</TableCell>
                <TableCell>Last Checked</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredResources
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((resource) => (
                  <TableRow key={resource.resourceId}>
                    <TableCell>{getStatusIcon(resource.complianceStatus)}</TableCell>
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {resource.resourceName}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {resource.resourceId}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip label={resource.resourceType} size="small" />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {resource.complianceScore.toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={resource.complianceScore}
                          sx={{
                            width: 60,
                            height: 6,
                            borderRadius: 3,
                            bgcolor: 'grey.200',
                          }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      {resource.violations.length > 0 ? (
                        <Chip
                          label={`${resource.violations.length} violations`}
                          color="error"
                          size="small"
                          variant="outlined"
                        />
                      ) : (
                        <Chip
                          label="No violations"
                          color="success"
                          size="small"
                          variant="outlined"
                        />
                      )}
                    </TableCell>
                    <TableCell>
                      {format(new Date(resource.lastChecked), 'MMM dd, HH:mm')}
                    </TableCell>
                    <TableCell>
                      <Tooltip title="View Details">
                        <IconButton size="small">
                          <Info />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={filteredResources.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={(_, newPage) => setPage(newPage)}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
        />
      </Paper>

      {/* Recommendations */}
      {metrics && metrics.criticalViolations > 0 && (
        <Alert severity="error" sx={{ mt: 3 }}>
          <AlertTitle>Critical Compliance Issues Detected</AlertTitle>
          You have {metrics.criticalViolations} critical violations that require
          immediate attention. Review and remediate these issues to improve your
          compliance posture.
        </Alert>
      )}
    </Box>
  );
};