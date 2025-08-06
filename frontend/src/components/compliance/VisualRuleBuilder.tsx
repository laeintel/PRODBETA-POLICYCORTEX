import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  NodeChange,
  EdgeChange,
  Connection,
  ReactFlowProvider,
  useReactFlow,
  Handle,
  Position,
  NodeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Box,
  Paper,
  Button,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Stack,
  Alert,
  Tabs,
  Tab,
  Card,
  CardContent,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  Add as AddIcon,
  Save as SaveIcon,
  PlayArrow as PlayIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Code as CodeIcon,
  Category as CategoryIcon,
  Rule as RuleIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  CompareArrows as CompareIcon,
  Functions as FunctionIcon,
  Schedule as ScheduleIcon,
  NotificationAdd as NotifyIcon,
  Block as BlockIcon,
  Build as BuildIcon,
  Label as LabelIcon,
  CloudUpload as UploadIcon,
  CloudDownload as DownloadIcon,
} from '@mui/icons-material';
import { api } from '../../services/api';

// Custom node components
const ConditionNode: React.FC<NodeProps> = ({ data, selected }) => {
  return (
    <Card
      sx={{
        minWidth: 200,
        border: selected ? '2px solid #1976d2' : '1px solid #ccc',
        borderRadius: 2,
        bgcolor: '#f5f5f5',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" spacing={1}>
          <CompareIcon color="primary" />
          <Typography variant="subtitle2" fontWeight="bold">
            {data.label}
          </Typography>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
          {data.description}
        </Typography>
        {data.field && (
          <Chip
            label={`Field: ${data.field}`}
            size="small"
            variant="outlined"
            sx={{ mt: 1 }}
          />
        )}
        {data.operator && (
          <Chip
            label={data.operator}
            size="small"
            color="primary"
            sx={{ mt: 0.5, ml: 0.5 }}
          />
        )}
      </CardContent>
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
    </Card>
  );
};

const ActionNode: React.FC<NodeProps> = ({ data, selected }) => {
  const getActionIcon = () => {
    switch (data.actionType) {
      case 'alert':
        return <WarningIcon color="warning" />;
      case 'remediate':
        return <BuildIcon color="success" />;
      case 'notify':
        return <NotifyIcon color="info" />;
      case 'block':
        return <BlockIcon color="error" />;
      case 'tag':
        return <LabelIcon color="primary" />;
      default:
        return <PlayIcon color="action" />;
    }
  };

  return (
    <Card
      sx={{
        minWidth: 200,
        border: selected ? '2px solid #4caf50' : '1px solid #ccc',
        borderRadius: 2,
        bgcolor: '#e8f5e9',
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" spacing={1}>
          {getActionIcon()}
          <Typography variant="subtitle2" fontWeight="bold">
            {data.label}
          </Typography>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
          {data.description}
        </Typography>
        {data.severity && (
          <Chip
            label={data.severity}
            size="small"
            color={
              data.severity === 'critical'
                ? 'error'
                : data.severity === 'high'
                ? 'warning'
                : 'default'
            }
            sx={{ mt: 1 }}
          />
        )}
      </CardContent>
      <Handle type="target" position={Position.Left} />
    </Card>
  );
};

const LogicalOperatorNode: React.FC<NodeProps> = ({ data, selected }) => {
  return (
    <Card
      sx={{
        minWidth: 100,
        border: selected ? '2px solid #ff9800' : '1px solid #ccc',
        borderRadius: '50%',
        bgcolor: '#fff3e0',
        textAlign: 'center',
      }}
    >
      <CardContent>
        <Typography variant="h6" fontWeight="bold">
          {data.operator}
        </Typography>
      </CardContent>
      <Handle type="source" position={Position.Right} />
      <Handle type="target" position={Position.Left} />
    </Card>
  );
};

const nodeTypes = {
  condition: ConditionNode,
  action: ActionNode,
  logical: LogicalOperatorNode,
};

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
}

interface RuleBuilderProps {
  onSave?: (rule: any) => void;
  initialRule?: any;
}

export const VisualRuleBuilder: React.FC<RuleBuilderProps> = ({
  onSave,
  initialRule,
}) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [selectedTab, setSelectedTab] = useState(0);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [validationResult, setValidationResult] = useState<any>(null);
  const [ruleMetadata, setRuleMetadata] = useState({
    name: '',
    description: '',
    severity: 'medium',
    tags: [] as string[],
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState('json');

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { project } = useReactFlow();

  useEffect(() => {
    initializeSession();
    loadTemplates();
  }, []);

  const initializeSession = async () => {
    try {
      const response = await api.post('/api/v1/rule-builder/sessions');
      setSessionId(response.data.session_id);
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const loadTemplates = async () => {
    try {
      const response = await api.get('/api/v1/rule-builder/templates');
      setTemplates(response.data);
    } catch (error) {
      console.error('Failed to load templates:', error);
    }
  };

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    []
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');
      const componentData = JSON.parse(
        event.dataTransfer.getData('componentData')
      );

      const position = project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node = {
        id: `${type}_${Date.now()}`,
        type,
        position,
        data: componentData,
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [project]
  );

  const handleDragStart = (
    event: React.DragEvent,
    nodeType: string,
    data: any
  ) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('componentData', JSON.stringify(data));
    event.dataTransfer.effectAllowed = 'move';
  };

  const validateRule = async () => {
    if (!sessionId) return;

    try {
      const response = await api.post(
        `/api/v1/rule-builder/sessions/${sessionId}/validate`
      );
      setValidationResult(response.data);
    } catch (error) {
      console.error('Validation failed:', error);
    }
  };

  const saveRule = async () => {
    if (!sessionId) return;

    try {
      await validateRule();
      
      if (validationResult && !validationResult.valid) {
        return;
      }

      const response = await api.post(
        `/api/v1/rule-builder/sessions/${sessionId}/save`
      );
      
      if (onSave) {
        onSave(response.data);
      }
    } catch (error) {
      console.error('Failed to save rule:', error);
    }
  };

  const exportRule = async () => {
    if (!sessionId) return;

    try {
      const response = await api.get(
        `/api/v1/rule-builder/sessions/${sessionId}/export`,
        { params: { format: exportFormat } }
      );
      
      // Download the exported rule
      const blob = new Blob([response.data.data], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `rule_${Date.now()}.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      setExportDialogOpen(false);
    } catch (error) {
      console.error('Failed to export rule:', error);
    }
  };

  const applyTemplate = async (templateId: string) => {
    if (!sessionId) return;

    try {
      await api.post(
        `/api/v1/rule-builder/sessions/${sessionId}/templates/${templateId}`
      );
      
      // Reload session to get updated components
      const response = await api.get(
        `/api/v1/rule-builder/sessions/${sessionId}`
      );
      
      // Convert to ReactFlow nodes
      const newNodes = response.data.components.map((comp: any, index: number) => ({
        id: comp.component_id,
        type: comp.type === 'condition' ? 'condition' : comp.type === 'action' ? 'action' : 'logical',
        position: comp.position || { x: 100 + index * 150, y: 100 + index * 50 },
        data: {
          label: comp.display_name,
          description: comp.properties.description || '',
          ...comp.properties,
        },
      }));
      
      setNodes(newNodes);
    } catch (error) {
      console.error('Failed to apply template:', error);
    }
  };

  const componentLibrary = {
    conditions: [
      {
        type: 'field_comparison',
        label: 'Field Comparison',
        icon: <CompareIcon />,
        description: 'Compare field values',
      },
      {
        type: 'custom_function',
        label: 'Custom Function',
        icon: <FunctionIcon />,
        description: 'Use pre-defined functions',
      },
      {
        type: 'date_range',
        label: 'Date Range',
        icon: <ScheduleIcon />,
        description: 'Check date ranges',
      },
    ],
    actions: [
      {
        type: 'alert',
        label: 'Create Alert',
        icon: <WarningIcon />,
        description: 'Generate an alert',
      },
      {
        type: 'remediate',
        label: 'Auto-Remediate',
        icon: <BuildIcon />,
        description: 'Automatically fix issues',
      },
      {
        type: 'notify',
        label: 'Send Notification',
        icon: <NotifyIcon />,
        description: 'Send notifications',
      },
      {
        type: 'block',
        label: 'Block Access',
        icon: <BlockIcon />,
        description: 'Block resource access',
      },
      {
        type: 'tag',
        label: 'Tag Resource',
        icon: <LabelIcon />,
        description: 'Add tags to resources',
      },
    ],
    operators: [
      {
        type: 'AND',
        label: 'AND',
        description: 'All conditions must match',
      },
      {
        type: 'OR',
        label: 'OR',
        description: 'Any condition must match',
      },
    ],
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* Component Library Drawer */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={drawerOpen}
        sx={{
          width: 300,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 300,
            boxSizing: 'border-box',
            position: 'relative',
            height: '100%',
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Rule Components
          </Typography>
          <Tabs value={selectedTab} onChange={(_, v) => setSelectedTab(v)}>
            <Tab label="Components" />
            <Tab label="Templates" />
          </Tabs>
        </Box>
        <Divider />
        
        {selectedTab === 0 ? (
          <Box sx={{ p: 2 }}>
            {/* Conditions */}
            <Typography variant="subtitle2" gutterBottom>
              Conditions
            </Typography>
            <List dense>
              {componentLibrary.conditions.map((item) => (
                <ListItem
                  key={item.type}
                  draggable
                  onDragStart={(e) =>
                    handleDragStart(e, 'condition', {
                      label: item.label,
                      description: item.description,
                      type: item.type,
                    })
                  }
                  sx={{
                    cursor: 'grab',
                    bgcolor: 'background.paper',
                    mb: 1,
                    borderRadius: 1,
                    border: '1px solid #e0e0e0',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText
                    primary={item.label}
                    secondary={item.description}
                  />
                </ListItem>
              ))}
            </List>

            <Divider sx={{ my: 2 }} />

            {/* Actions */}
            <Typography variant="subtitle2" gutterBottom>
              Actions
            </Typography>
            <List dense>
              {componentLibrary.actions.map((item) => (
                <ListItem
                  key={item.type}
                  draggable
                  onDragStart={(e) =>
                    handleDragStart(e, 'action', {
                      label: item.label,
                      description: item.description,
                      actionType: item.type,
                    })
                  }
                  sx={{
                    cursor: 'grab',
                    bgcolor: 'background.paper',
                    mb: 1,
                    borderRadius: 1,
                    border: '1px solid #e0e0e0',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText
                    primary={item.label}
                    secondary={item.description}
                  />
                </ListItem>
              ))}
            </List>

            <Divider sx={{ my: 2 }} />

            {/* Logical Operators */}
            <Typography variant="subtitle2" gutterBottom>
              Logical Operators
            </Typography>
            <List dense>
              {componentLibrary.operators.map((item) => (
                <ListItem
                  key={item.type}
                  draggable
                  onDragStart={(e) =>
                    handleDragStart(e, 'logical', {
                      label: item.label,
                      operator: item.type,
                    })
                  }
                  sx={{
                    cursor: 'grab',
                    bgcolor: 'background.paper',
                    mb: 1,
                    borderRadius: 1,
                    border: '1px solid #e0e0e0',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <ListItemText
                    primary={item.label}
                    secondary={item.description}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        ) : (
          <Box sx={{ p: 2 }}>
            {/* Templates */}
            <List>
              {templates.map((template) => (
                <ListItem
                  key={template.id}
                  button
                  onClick={() => applyTemplate(template.id)}
                  sx={{
                    mb: 1,
                    borderRadius: 1,
                    border: '1px solid #e0e0e0',
                  }}
                >
                  <ListItemIcon>
                    <CategoryIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary={template.name}
                    secondary={template.description}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Drawer>

      {/* Main Canvas */}
      <Box sx={{ flexGrow: 1, position: 'relative' }}>
        {/* Toolbar */}
        <Paper
          sx={{
            position: 'absolute',
            top: 16,
            left: 16,
            right: 16,
            zIndex: 10,
            p: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <IconButton
            onClick={() => setDrawerOpen(!drawerOpen)}
            size="small"
          >
            <CategoryIcon />
          </IconButton>
          
          <TextField
            size="small"
            label="Rule Name"
            value={ruleMetadata.name}
            onChange={(e) =>
              setRuleMetadata({ ...ruleMetadata, name: e.target.value })
            }
            sx={{ minWidth: 200 }}
          />
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Severity</InputLabel>
            <Select
              value={ruleMetadata.severity}
              label="Severity"
              onChange={(e) =>
                setRuleMetadata({
                  ...ruleMetadata,
                  severity: e.target.value,
                })
              }
            >
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
            </Select>
          </FormControl>

          <Box sx={{ flexGrow: 1 }} />

          {validationResult && (
            <Chip
              icon={
                validationResult.valid ? (
                  <CheckIcon />
                ) : (
                  <ErrorIcon />
                )
              }
              label={
                validationResult.valid
                  ? 'Valid'
                  : `${validationResult.errors?.length || 0} errors`
              }
              color={validationResult.valid ? 'success' : 'error'}
              variant="outlined"
            />
          )}

          <Button
            startIcon={<PlayIcon />}
            variant="outlined"
            onClick={validateRule}
          >
            Validate
          </Button>
          
          <Button
            startIcon={<SaveIcon />}
            variant="contained"
            onClick={saveRule}
          >
            Save Rule
          </Button>
          
          <IconButton onClick={() => setExportDialogOpen(true)}>
            <DownloadIcon />
          </IconButton>
          
          <IconButton onClick={() => setSettingsOpen(true)}>
            <SettingsIcon />
          </IconButton>
        </Paper>

        {/* Validation Messages */}
        {validationResult && !validationResult.valid && (
          <Alert
            severity="error"
            sx={{
              position: 'absolute',
              bottom: 16,
              left: 16,
              right: 16,
              zIndex: 10,
            }}
            onClose={() => setValidationResult(null)}
          >
            <Stack spacing={1}>
              {validationResult.errors?.map((error: string, index: number) => (
                <Typography key={index} variant="body2">
                  • {error}
                </Typography>
              ))}
            </Stack>
          </Alert>
        )}

        {/* ReactFlow Canvas */}
        <div ref={reactFlowWrapper} style={{ width: '100%', height: '100%' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <Background variant="dots" gap={12} size={1} />
          </ReactFlow>
        </div>
      </Box>

      {/* Settings Dialog */}
      <Dialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Rule Settings</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Rule Name"
              value={ruleMetadata.name}
              onChange={(e) =>
                setRuleMetadata({ ...ruleMetadata, name: e.target.value })
              }
            />
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Description"
              value={ruleMetadata.description}
              onChange={(e) =>
                setRuleMetadata({
                  ...ruleMetadata,
                  description: e.target.value,
                })
              }
            />
            <FormControl fullWidth>
              <InputLabel>Severity</InputLabel>
              <Select
                value={ruleMetadata.severity}
                label="Severity"
                onChange={(e) =>
                  setRuleMetadata({
                    ...ruleMetadata,
                    severity: e.target.value,
                  })
                }
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button
            onClick={() => setSettingsOpen(false)}
            variant="contained"
          >
            Save Settings
          </Button>
        </DialogActions>
      </Dialog>

      {/* Export Dialog */}
      <Dialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Export Rule</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Format</InputLabel>
            <Select
              value={exportFormat}
              label="Format"
              onChange={(e) => setExportFormat(e.target.value)}
            >
              <MenuItem value="json">JSON</MenuItem>
              <MenuItem value="yaml">YAML</MenuItem>
              <MenuItem value="python">Python Code</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
          <Button onClick={exportRule} variant="contained">
            Export
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Wrap component with ReactFlowProvider
export default function VisualRuleBuilderWrapper(props: RuleBuilderProps) {
  return (
    <ReactFlowProvider>
      <VisualRuleBuilder {...props} />
    </ReactFlowProvider>
  );
}