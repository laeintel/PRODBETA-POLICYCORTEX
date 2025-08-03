import React, { useState, useEffect } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material'
import {
  Close as CloseIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Security as SecurityIcon,
  Build as BuildIcon,
  Schedule as ScheduleIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayArrowIcon
} from '@mui/icons-material'

interface ResourceComplianceDetails {
  resourceId: string
  resourceName: string
  resourceType: string
  resourceGroup: string
  location: string
  compliance: {
    status: string
    totalPolicies: number
    compliantPolicies: number
    violatingPolicies: number
    complianceScore: number
    lastEvaluated: string
  }
  violations: Array<{
    policyName: string
    policyDefinitionId: string
    effect: string
    description: string
    severity: string
    evaluatedOn: string
    reason: string
  }>
  remediationSteps: Array<{
    step: number
    title: string
    description: string
    action: string
    automated: boolean
    estimatedTime: string
  }>
  appliedPolicies: Array<{
    name: string
    type: string
    effect: string
    status: string
  }>
  recommendations: Array<{
    priority: string
    title: string
    description: string
    category: string
  }>
}

interface ResourceDetailModalProps {
  open: boolean
  onClose: () => void
  resourceId: string
  resourceName: string
}

const ResourceDetailModal: React.FC<ResourceDetailModalProps> = ({
  open,
  onClose,
  resourceId,
  resourceName
}) => {
  const [details, setDetails] = useState<ResourceComplianceDetails | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    if (open && resourceId) {
      fetchResourceDetails()
    }
  }, [open, resourceId])

  const fetchResourceDetails = async () => {
    setLoading(true)
    setError(null)
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
      const response = await fetch(`${apiBaseUrl}/api/v1/resources/${encodeURIComponent(resourceId)}/compliance`)
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data = await response.json()
      setDetails(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch resource details')
    } finally {
      setLoading(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'error'
      case 'medium': return 'warning'
      case 'low': return 'info'
      default: return 'default'
    }
  }

  const getComplianceIcon = (status: string) => {
    return status === 'Compliant' ? (
      <CheckCircleIcon color="success" />
    ) : (
      <ErrorIcon color="error" />
    )
  }

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high': return 'error'
      case 'medium': return 'warning'
      case 'low': return 'info'
      default: return 'default'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  if (loading) {
    return (
      <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
        <DialogContent sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
          <CircularProgress />
        </DialogContent>
      </Dialog>
    )
  }

  if (error) {
    return (
      <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
        <DialogTitle>Error</DialogTitle>
        <DialogContent>
          <Alert severity="error">{error}</Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Close</Button>
        </DialogActions>
      </Dialog>
    )
  }

  if (!details) {
    return null
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6">Resource Compliance Details</Typography>
            <Typography variant="subtitle2" color="text.secondary">
              {details.resourceName}
            </Typography>
          </Box>
          <Button onClick={onClose} startIcon={<CloseIcon />}>
            Close
          </Button>
        </Box>
      </DialogTitle>
      
      <DialogContent dividers>
        <Grid container spacing={3}>
          {/* Resource Overview */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Resource Overview
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Type</Typography>
                    <Typography variant="body1">{details.resourceType}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Resource Group</Typography>
                    <Typography variant="body1">{details.resourceGroup}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Location</Typography>
                    <Typography variant="body1">{details.location}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Compliance Status</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getComplianceIcon(details.compliance.status)}
                      <Chip 
                        label={details.compliance.status}
                        color={details.compliance.status === 'Compliant' ? 'success' : 'error'}
                        size="small"
                      />
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Compliance Summary */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Compliance Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="h4" color="primary">{details.compliance.complianceScore}%</Typography>
                    <Typography variant="body2" color="text.secondary">Compliance Score</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="h4" color="text.primary">{details.compliance.totalPolicies}</Typography>
                    <Typography variant="body2" color="text.secondary">Total Policies</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body1" color="success.main">{details.compliance.compliantPolicies}</Typography>
                    <Typography variant="body2" color="text.secondary">Compliant</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body1" color="error.main">{details.compliance.violatingPolicies}</Typography>
                    <Typography variant="body2" color="text.secondary">Violations</Typography>
                  </Grid>
                </Grid>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Last evaluated: {formatDate(details.compliance.lastEvaluated)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Applied Policies */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Applied Policies
                </Typography>
                <List dense>
                  {details.appliedPolicies.map((policy, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {getComplianceIcon(policy.status)}
                      </ListItemIcon>
                      <ListItemText
                        primary={policy.name}
                        secondary={`${policy.type} - ${policy.effect}`}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Violations */}
          {details.violations.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <WarningIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Policy Violations ({details.violations.length})
                  </Typography>
                  {details.violations.map((violation, index) => (
                    <Accordion key={index} sx={{ mb: 1 }}>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                          <Chip
                            label={violation.severity}
                            color={getSeverityColor(violation.severity) as any}
                            size="small"
                          />
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            {violation.policyName}
                          </Typography>
                          <Chip label={violation.effect} size="small" variant="outlined" />
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="body2" paragraph>
                          {violation.description}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Reason:</strong> {violation.reason}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          Evaluated on: {formatDate(violation.evaluatedOn)}
                        </Typography>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Remediation Steps */}
          {details.remediationSteps.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <BuildIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Remediation Steps
                  </Typography>
                  <Stepper activeStep={activeStep} orientation="vertical">
                    {details.remediationSteps.map((step, index) => (
                      <Step key={index}>
                        <StepLabel
                          optional={
                            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                              <Chip
                                label={step.automated ? 'Automated' : 'Manual'}
                                color={step.automated ? 'success' : 'warning'}
                                size="small"
                              />
                              <Chip
                                label={step.estimatedTime}
                                icon={<ScheduleIcon />}
                                size="small"
                                variant="outlined"
                              />
                            </Box>
                          }
                        >
                          {step.title}
                        </StepLabel>
                        <StepContent>
                          <Typography variant="body2" paragraph>
                            {step.description}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" paragraph>
                            <strong>Action:</strong> {step.action}
                          </Typography>
                          <Box sx={{ mt: 2 }}>
                            <Button
                              variant="outlined"
                              size="small"
                              startIcon={<PlayArrowIcon />}
                              disabled={!step.automated}
                            >
                              {step.automated ? 'Execute' : 'Manual Action Required'}
                            </Button>
                          </Box>
                        </StepContent>
                      </Step>
                    ))}
                  </Stepper>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Recommendations */}
          {details.recommendations.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recommendations
                  </Typography>
                  {details.recommendations.map((rec, index) => (
                    <Alert 
                      key={index} 
                      severity={getPriorityColor(rec.priority) as any}
                      sx={{ mb: 1 }}
                    >
                      <Typography variant="subtitle2">{rec.title}</Typography>
                      <Typography variant="body2">{rec.description}</Typography>
                      <Chip label={rec.category} size="small" sx={{ mt: 1 }} />
                    </Alert>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} variant="outlined">
          Close
        </Button>
        {details.violations.length > 0 && (
          <Button variant="contained" color="primary">
            Start Remediation
          </Button>
        )}
      </DialogActions>
    </Dialog>
  )
}

export default ResourceDetailModal