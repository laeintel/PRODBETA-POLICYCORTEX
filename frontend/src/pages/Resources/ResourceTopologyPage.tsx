import React, { useState, useEffect, useMemo, useRef } from 'react'
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Grid,
  Button,
  Stack,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse
} from '@mui/material'
import {
  AccountTreeOutlined,
  RefreshOutlined,
  FullscreenOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  CenterFocusStrongOutlined,
  LayersOutlined,
  StorageOutlined,
  ComputerOutlined,
  NetworkPingOutlined,
  SecurityOutlined,
  CloudOutlined,
  ExpandMoreOutlined,
  ExpandLessOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { useFilter } from '../../contexts/FilterContext'
import GlobalFilterPanel from '../../components/Filters/GlobalFilterPanel'
import { apiClient } from '../../services/apiClient'

interface TopologyNode {
  id: string
  name: string
  type: string
  resourceGroup: string
  subscription: string
  location: string
  status: string
  dependencies: string[]
  dependents: string[]
  position: { x: number; y: number }
  size: number
  color: string
}

interface TopologyEdge {
  id: string
  source: string
  target: string
  type: 'dependency' | 'network' | 'data'
  label?: string
}

interface TopologyData {
  nodes: TopologyNode[]
  edges: TopologyEdge[]
  summary: {
    totalNodes: number
    totalEdges: number
    resourceGroups: number
    subscriptions: number
  }
  data_source: string
}

const ResourceTopologyPage = () => {
  const { applyFilters } = useFilter()
  const svgRef = useRef<SVGSVGElement>(null)
  const [topologyData, setTopologyData] = useState<TopologyData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<TopologyNode | null>(null)
  const [viewMode, setViewMode] = useState<'logical' | 'physical'>('logical')
  const [showLabels, setShowLabels] = useState(true)
  const [showDependencies, setShowDependencies] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [legendExpanded, setLegendExpanded] = useState(false)

  const fetchTopologyData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Start with mock data immediately
      setTopologyData(generateMockTopologyData())
      setLoading(false)
      
      // Try to fetch real data in background
      try {
        const response = await apiClient.get('/api/v1/resources/topology', { timeout: 5000 })
        if (response?.data) {
          setTopologyData(response.data)
        }
      } catch (backgroundErr) {
        console.log('[API] Background topology fetch failed, keeping mock data')
      }
    } catch (err: any) {
      console.error('Error fetching topology data:', err)
      setError('Using sample topology data')
      setTopologyData(generateMockTopologyData())
      setLoading(false)
    }
  }

  const generateMockTopologyData = (): TopologyData => {
    const nodes: TopologyNode[] = [
      // Subscription level
      {
        id: 'sub-1',
        name: 'PolicyCortex Prod',
        type: 'subscription',
        resourceGroup: '',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'Global',
        status: 'Active',
        dependencies: [],
        dependents: ['rg-prod', 'rg-shared'],
        position: { x: 400, y: 50 },
        size: 40,
        color: '#1976d2'
      },
      // Resource Groups
      {
        id: 'rg-prod',
        name: 'rg-policycortex-prod',
        type: 'resourceGroup',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Active',
        dependencies: ['sub-1'],
        dependents: ['aks-prod', 'sql-prod', 'storage-prod'],
        position: { x: 200, y: 150 },
        size: 35,
        color: '#ff9800'
      },
      {
        id: 'rg-shared',
        name: 'rg-policycortex-shared',
        type: 'resourceGroup',
        resourceGroup: 'rg-policycortex-shared',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Active',
        dependencies: ['sub-1'],
        dependents: ['acr-shared', 'kv-shared'],
        position: { x: 600, y: 150 },
        size: 35,
        color: '#ff9800'
      },
      // Compute Resources
      {
        id: 'aks-prod',
        name: 'aks-policycortex-prod',
        type: 'Microsoft.ContainerService/managedClusters',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Running',
        dependencies: ['rg-prod', 'vnet-prod', 'acr-shared'],
        dependents: ['app-prod-1', 'app-prod-2'],
        position: { x: 100, y: 250 },
        size: 30,
        color: '#4caf50'
      },
      // Applications
      {
        id: 'app-prod-1',
        name: 'api-gateway-prod',
        type: 'Microsoft.ContainerInstance/containerGroups',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Running',
        dependencies: ['aks-prod', 'sql-prod'],
        dependents: [],
        position: { x: 50, y: 350 },
        size: 25,
        color: '#2196f3'
      },
      {
        id: 'app-prod-2',
        name: 'ai-engine-prod',
        type: 'Microsoft.ContainerInstance/containerGroups',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Running',
        dependencies: ['aks-prod', 'storage-prod'],
        dependents: [],
        position: { x: 150, y: 350 },
        size: 25,
        color: '#2196f3'
      },
      // Data Resources
      {
        id: 'sql-prod',
        name: 'policycortex-sql-prod',
        type: 'Microsoft.Sql/servers',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Online',
        dependencies: ['rg-prod', 'vnet-prod'],
        dependents: ['app-prod-1'],
        position: { x: 300, y: 250 },
        size: 30,
        color: '#f44336'
      },
      // Storage
      {
        id: 'storage-prod',
        name: 'stpolicycortexprod',
        type: 'Microsoft.Storage/storageAccounts',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Available',
        dependencies: ['rg-prod'],
        dependents: ['app-prod-2'],
        position: { x: 200, y: 350 },
        size: 25,
        color: '#9c27b0'
      },
      // Network
      {
        id: 'vnet-prod',
        name: 'vnet-policycortex-prod',
        type: 'Microsoft.Network/virtualNetworks',
        resourceGroup: 'rg-policycortex-prod',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Available',
        dependencies: ['rg-prod'],
        dependents: ['aks-prod', 'sql-prod'],
        position: { x: 250, y: 180 },
        size: 28,
        color: '#607d8b'
      },
      // Shared Resources
      {
        id: 'acr-shared',
        name: 'policycortexacr',
        type: 'Microsoft.ContainerRegistry/registries',
        resourceGroup: 'rg-policycortex-shared',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Available',
        dependencies: ['rg-shared'],
        dependents: ['aks-prod'],
        position: { x: 550, y: 250 },
        size: 25,
        color: '#795548'
      },
      {
        id: 'kv-shared',
        name: 'kv-policycortex-shared',
        type: 'Microsoft.KeyVault/vaults',
        resourceGroup: 'rg-policycortex-shared',
        subscription: '9f16cc88-89ce-49ba-a96d-308ed3169595',
        location: 'East US',
        status: 'Available',
        dependencies: ['rg-shared'],
        dependents: ['app-prod-1', 'app-prod-2'],
        position: { x: 650, y: 250 },
        size: 25,
        color: '#ff5722'
      }
    ]

    const edges: TopologyEdge[] = [
      // Subscription to Resource Groups
      { id: 'e1', source: 'sub-1', target: 'rg-prod', type: 'dependency' },
      { id: 'e2', source: 'sub-1', target: 'rg-shared', type: 'dependency' },
      // Resource Group to Resources
      { id: 'e3', source: 'rg-prod', target: 'aks-prod', type: 'dependency' },
      { id: 'e4', source: 'rg-prod', target: 'sql-prod', type: 'dependency' },
      { id: 'e5', source: 'rg-prod', target: 'storage-prod', type: 'dependency' },
      { id: 'e6', source: 'rg-prod', target: 'vnet-prod', type: 'dependency' },
      { id: 'e7', source: 'rg-shared', target: 'acr-shared', type: 'dependency' },
      { id: 'e8', source: 'rg-shared', target: 'kv-shared', type: 'dependency' },
      // Application Dependencies
      { id: 'e9', source: 'aks-prod', target: 'app-prod-1', type: 'dependency' },
      { id: 'e10', source: 'aks-prod', target: 'app-prod-2', type: 'dependency' },
      { id: 'e11', source: 'app-prod-1', target: 'sql-prod', type: 'data', label: 'Database' },
      { id: 'e12', source: 'app-prod-2', target: 'storage-prod', type: 'data', label: 'Files' },
      { id: 'e13', source: 'aks-prod', target: 'acr-shared', type: 'dependency', label: 'Images' },
      { id: 'e14', source: 'app-prod-1', target: 'kv-shared', type: 'dependency', label: 'Secrets' },
      { id: 'e15', source: 'app-prod-2', target: 'kv-shared', type: 'dependency', label: 'Secrets' },
      // Network Dependencies
      { id: 'e16', source: 'vnet-prod', target: 'aks-prod', type: 'network' },
      { id: 'e17', source: 'vnet-prod', target: 'sql-prod', type: 'network' }
    ]

    return {
      nodes,
      edges,
      summary: {
        totalNodes: nodes.length,
        totalEdges: edges.length,
        resourceGroups: new Set(nodes.map(n => n.resourceGroup).filter(Boolean)).size,
        subscriptions: new Set(nodes.map(n => n.subscription).filter(Boolean)).size
      },
      data_source: 'mock-topology-data'
    }
  }

  // Apply filters to topology data
  const filteredTopology = useMemo(() => {
    if (!topologyData) return null
    
    const filteredNodes = applyFilters(topologyData.nodes)
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id))
    const filteredEdges = topologyData.edges.filter(edge => 
      filteredNodeIds.has(edge.source) && filteredNodeIds.has(edge.target)
    )
    
    return {
      ...topologyData,
      nodes: filteredNodes,
      edges: filteredEdges
    }
  }, [topologyData, applyFilters])

  useEffect(() => {
    fetchTopologyData()
  }, [])

  const getResourceIcon = (type: string) => {
    const lowerType = type.toLowerCase()
    if (lowerType.includes('storage')) return <StorageOutlined />
    if (lowerType.includes('compute') || lowerType.includes('container')) return <ComputerOutlined />
    if (lowerType.includes('network')) return <NetworkPingOutlined />
    if (lowerType.includes('keyvault') || lowerType.includes('security')) return <SecurityOutlined />
    return <CloudOutlined />
  }

  const handleNodeClick = (node: TopologyNode) => {
    setSelectedNode(node)
  }

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.2, 3))
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.2, 0.2))
  }

  const handleResetView = () => {
    setZoom(1)
  }

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <>
      <Helmet>
        <title>Resource Topology - PolicyCortex</title>
        <meta name="description" content="Interactive infrastructure topology and dependencies" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AccountTreeOutlined />
            Infrastructure Topology
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<RefreshOutlined />}
              onClick={fetchTopologyData}
              disabled={loading}
            >
              Refresh
            </Button>
          </Stack>
        </Box>

        {error && (
          <Alert severity="info" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Global Filter Panel */}
        <GlobalFilterPanel
          availableResourceGroups={topologyData?.nodes.map(n => n.resourceGroup).filter(Boolean) || []}
          availableResourceTypes={topologyData?.nodes.map(n => n.type) || []}
        />

        {filteredTopology && (
          <>
            {/* Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <AccountTreeOutlined color="primary" />
                      <Box>
                        <Typography variant="h4">{filteredTopology.nodes.length}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Resources
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <NetworkPingOutlined color="info" />
                      <Box>
                        <Typography variant="h4">{filteredTopology.edges.length}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Dependencies
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <LayersOutlined color="secondary" />
                      <Box>
                        <Typography variant="h4">
                          {new Set(filteredTopology.nodes.map(n => n.resourceGroup).filter(Boolean)).size}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Resource Groups
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <CloudOutlined color="warning" />
                      <Box>
                        <Typography variant="h4">
                          {new Set(filteredTopology.nodes.map(n => n.subscription).filter(Boolean)).size}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Subscriptions
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Grid container spacing={3}>
              {/* Topology Visualization */}
              <Grid item xs={12} lg={9}>
                <Paper sx={{ overflow: 'hidden', position: 'relative' }}>
                  {/* Toolbar */}
                  <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6">Infrastructure Map</Typography>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <FormControlLabel
                        control={
                          <Switch
                            checked={showLabels}
                            onChange={(e) => setShowLabels(e.target.checked)}
                            size="small"
                          />
                        }
                        label="Labels"
                      />
                      <FormControlLabel
                        control={
                          <Switch
                            checked={showDependencies}
                            onChange={(e) => setShowDependencies(e.target.checked)}
                            size="small"
                          />
                        }
                        label="Dependencies"
                      />
                      <Divider orientation="vertical" flexItem />
                      <Tooltip title="Zoom In">
                        <IconButton size="small" onClick={handleZoomIn}>
                          <ZoomInOutlined />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Zoom Out">
                        <IconButton size="small" onClick={handleZoomOut}>
                          <ZoomOutOutlined />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Reset View">
                        <IconButton size="small" onClick={handleResetView}>
                          <CenterFocusStrongOutlined />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Fullscreen">
                        <IconButton size="small">
                          <FullscreenOutlined />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  </Box>

                  {/* SVG Topology */}
                  <Box sx={{ height: 600, position: 'relative', overflow: 'hidden' }}>
                    <svg
                      ref={svgRef}
                      width="100%"
                      height="100%"
                      viewBox={`0 0 800 600`}
                      style={{ transform: `scale(${zoom})`, transformOrigin: 'center' }}
                    >
                      {/* Background Grid */}
                      <defs>
                        <pattern
                          id="grid"
                          width="40"
                          height="40"
                          patternUnits="userSpaceOnUse"
                        >
                          <path
                            d="M 40 0 L 0 0 0 40"
                            fill="none"
                            stroke="#e0e0e0"
                            strokeWidth="1"
                            opacity="0.3"
                          />
                        </pattern>
                      </defs>
                      <rect width="100%" height="100%" fill="url(#grid)" />

                      {/* Edges */}
                      {showDependencies && filteredTopology.edges.map((edge) => {
                        const sourceNode = filteredTopology.nodes.find(n => n.id === edge.source)
                        const targetNode = filteredTopology.nodes.find(n => n.id === edge.target)
                        if (!sourceNode || !targetNode) return null

                        const strokeColor = edge.type === 'dependency' ? '#1976d2' : 
                                          edge.type === 'network' ? '#607d8b' : '#4caf50'
                        const strokeDasharray = edge.type === 'data' ? '5,5' : '0'

                        return (
                          <g key={edge.id}>
                            <line
                              x1={sourceNode.position.x}
                              y1={sourceNode.position.y}
                              x2={targetNode.position.x}
                              y2={targetNode.position.y}
                              stroke={strokeColor}
                              strokeWidth="2"
                              strokeDasharray={strokeDasharray}
                              opacity="0.6"
                              markerEnd="url(#arrowhead)"
                            />
                            {showLabels && edge.label && (
                              <text
                                x={(sourceNode.position.x + targetNode.position.x) / 2}
                                y={(sourceNode.position.y + targetNode.position.y) / 2 - 5}
                                fontSize="10"
                                fill="#666"
                                textAnchor="middle"
                                style={{ pointerEvents: 'none' }}
                              >
                                {edge.label}
                              </text>
                            )}
                          </g>
                        )
                      })}

                      {/* Arrow marker */}
                      <defs>
                        <marker
                          id="arrowhead"
                          markerWidth="10"
                          markerHeight="7"
                          refX="9"
                          refY="3.5"
                          orient="auto"
                        >
                          <polygon
                            points="0 0, 10 3.5, 0 7"
                            fill="#666"
                            opacity="0.6"
                          />
                        </marker>
                      </defs>

                      {/* Nodes */}
                      {filteredTopology.nodes.map((node) => (
                        <g key={node.id}>
                          <circle
                            cx={node.position.x}
                            cy={node.position.y}
                            r={node.size}
                            fill={node.color}
                            stroke={selectedNode?.id === node.id ? '#ff9800' : '#fff'}
                            strokeWidth={selectedNode?.id === node.id ? 3 : 2}
                            style={{ cursor: 'pointer' }}
                            onClick={() => handleNodeClick(node)}
                          />
                          {/* Status indicator */}
                          <circle
                            cx={node.position.x + node.size * 0.6}
                            cy={node.position.y - node.size * 0.6}
                            r="4"
                            fill={node.status === 'Running' || node.status === 'Available' || node.status === 'Online' || node.status === 'Active' ? '#4caf50' : '#f44336'}
                            stroke="#fff"
                            strokeWidth="1"
                          />
                          {showLabels && (
                            <text
                              x={node.position.x}
                              y={node.position.y + node.size + 15}
                              fontSize="12"
                              fill="#333"
                              textAnchor="middle"
                              style={{ pointerEvents: 'none' }}
                            >
                              {node.name.length > 20 ? node.name.substring(0, 20) + '...' : node.name}
                            </text>
                          )}
                        </g>
                      ))}
                    </svg>
                  </Box>
                </Paper>
              </Grid>

              {/* Sidebar */}
              <Grid item xs={12} lg={3}>
                <Stack spacing={3}>
                  {/* Legend */}
                  <Paper sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6">Legend</Typography>
                      <IconButton size="small" onClick={() => setLegendExpanded(!legendExpanded)}>
                        {legendExpanded ? <ExpandLessOutlined /> : <ExpandMoreOutlined />}
                      </IconButton>
                    </Box>
                    <Collapse in={legendExpanded}>
                      <Stack spacing={1}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#1976d2' }} />
                          <Typography variant="body2">Subscription</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#ff9800' }} />
                          <Typography variant="body2">Resource Group</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#4caf50' }} />
                          <Typography variant="body2">Compute</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#f44336' }} />
                          <Typography variant="body2">Database</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#9c27b0' }} />
                          <Typography variant="body2">Storage</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#607d8b' }} />
                          <Typography variant="body2">Network</Typography>
                        </Box>
                      </Stack>
                    </Collapse>
                  </Paper>

                  {/* Selected Node Details */}
                  {selectedNode && (
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" gutterBottom>
                        Resource Details
                      </Typography>
                      <List dense>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemIcon>
                            {getResourceIcon(selectedNode.type)}
                          </ListItemIcon>
                          <ListItemText
                            primary={selectedNode.name}
                            secondary={selectedNode.type.split('/').pop()}
                          />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText
                            primary="Resource Group"
                            secondary={selectedNode.resourceGroup || 'N/A'}
                          />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText
                            primary="Location"
                            secondary={selectedNode.location}
                          />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText
                            primary="Status"
                            secondary={
                              <Chip
                                label={selectedNode.status}
                                size="small"
                                color={selectedNode.status === 'Running' || selectedNode.status === 'Available' || selectedNode.status === 'Online' || selectedNode.status === 'Active' ? 'success' : 'error'}
                              />
                            }
                          />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText
                            primary="Dependencies"
                            secondary={`${selectedNode.dependencies.length} resources`}
                          />
                        </ListItem>
                        <ListItem sx={{ px: 0 }}>
                          <ListItemText
                            primary="Dependents"
                            secondary={`${selectedNode.dependents.length} resources`}
                          />
                        </ListItem>
                      </List>
                    </Paper>
                  )}
                </Stack>
              </Grid>
            </Grid>

            {/* Data Source Info */}
            <Box sx={{ mt: 3, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Data source: {filteredTopology.data_source} • Last updated: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </>
  )
}

export default ResourceTopologyPage