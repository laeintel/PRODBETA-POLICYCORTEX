import React, { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  TextField,
  Autocomplete,
  Chip,
  Button,
  Grid,
  Collapse,
  IconButton,
  Divider,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  OutlinedInput
} from '@mui/material'
import {
  FilterListOutlined,
  ExpandMoreOutlined,
  ExpandLessOutlined,
  ClearOutlined,
  SearchOutlined
} from '@mui/icons-material'
import { useFilter } from '../../contexts/FilterContext'

interface GlobalFilterPanelProps {
  availableSubscriptions?: string[]
  availableResourceGroups?: string[]
  availableResourceTypes?: string[]
  availableLocations?: string[]
  showAdvanced?: boolean
}

const GlobalFilterPanel: React.FC<GlobalFilterPanelProps> = ({
  availableSubscriptions = [],
  availableResourceGroups = [],
  availableResourceTypes = [],
  availableLocations = [],
  showAdvanced = true
}) => {
  const { filters, setFilters, resetFilters, isFilterActive } = useFilter()
  const [expanded, setExpanded] = useState(false)
  const [advancedExpanded, setAdvancedExpanded] = useState(false)

  // Default options if not provided
  const defaultSubscriptions = [
    '205b477d-17e7-4b3b-92c1-32cf02626b78',
    '9f16cc88-89ce-49ba-a96d-308ed3169595',
    '2ed94599-4ae6-415e-990e-2c66a3d4f9c3',
    '632a3b06-2a6c-4b07-8f4f-6bf4c6184095'
  ]

  const defaultResourceGroups = [
    'rg-policycortex-prod',
    'rg-policycortex-dev',
    'rg-policycortex-staging',
    'rg-policycortex-shared',
    'default-rg',
    'network-rg'
  ]

  const defaultResourceTypes = [
    'Microsoft.Compute/virtualMachines',
    'Microsoft.Storage/storageAccounts',
    'Microsoft.ContainerRegistry/registries',
    'Microsoft.Sql/servers',
    'Microsoft.Network/virtualNetworks',
    'Microsoft.Web/sites',
    'Microsoft.ContainerInstance/containerGroups',
    'Microsoft.KeyVault/vaults',
    'Microsoft.Insights/components'
  ]

  const defaultLocations = [
    'East US',
    'West US 2',
    'Central US',
    'North Europe',
    'West Europe',
    'Southeast Asia'
  ]

  const subscriptionOptions = availableSubscriptions.length > 0 ? availableSubscriptions : defaultSubscriptions
  const resourceGroupOptions = availableResourceGroups.length > 0 ? availableResourceGroups : defaultResourceGroups
  const resourceTypeOptions = availableResourceTypes.length > 0 ? availableResourceTypes : defaultResourceTypes
  const locationOptions = availableLocations.length > 0 ? availableLocations : defaultLocations

  const handleFilterChange = (field: keyof typeof filters, value: any) => {
    setFilters({ [field]: value })
  }

  return (
    <Paper sx={{ mb: 3, overflow: 'hidden' }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
           onClick={() => setExpanded(!expanded)}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FilterListOutlined color="primary" />
          <Typography variant="h6">Filters</Typography>
          {isFilterActive && (
            <Chip 
              label="Active" 
              size="small" 
              color="primary" 
              variant="outlined"
            />
          )}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isFilterActive && (
            <Button
              size="small"
              startIcon={<ClearOutlined />}
              onClick={(e) => {
                e.stopPropagation()
                resetFilters()
              }}
              color="error"
              variant="outlined"
            >
              Clear All
            </Button>
          )}
          <IconButton>
            {expanded ? <ExpandLessOutlined /> : <ExpandMoreOutlined />}
          </IconButton>
        </Box>
      </Box>

      <Collapse in={expanded}>
        <Divider />
        <Box sx={{ p: 3 }}>
          <Grid container spacing={3}>
            {/* Resource Name Search */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Resource Name"
                placeholder="Search by resource name..."
                value={filters.resourceName}
                onChange={(e) => handleFilterChange('resourceName', e.target.value)}
                InputProps={{
                  startAdornment: <SearchOutlined sx={{ mr: 1, color: 'text.secondary' }} />
                }}
                helperText="Supports partial matching"
              />
            </Grid>

            {/* Resource Type */}
            <Grid item xs={12} md={6}>
              <Autocomplete
                multiple
                options={resourceTypeOptions}
                value={filters.resourceTypes}
                onChange={(_, value) => handleFilterChange('resourceTypes', value)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Resource Type"
                    placeholder="Select resource types..."
                  />
                )}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip
                      variant="outlined"
                      label={option.split('/').pop()}
                      {...getTagProps({ index })}
                      key={option}
                    />
                  ))
                }
              />
            </Grid>

            {/* Subscription */}
            <Grid item xs={12} md={6}>
              <Autocomplete
                multiple
                options={subscriptionOptions}
                value={filters.subscriptions}
                onChange={(_, value) => handleFilterChange('subscriptions', value)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Subscription"
                    placeholder="Select subscriptions..."
                  />
                )}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip
                      variant="outlined"
                      label={option.substring(0, 8) + '...'}
                      {...getTagProps({ index })}
                      key={option}
                    />
                  ))
                }
              />
            </Grid>

            {/* Resource Group */}
            <Grid item xs={12} md={6}>
              <Autocomplete
                multiple
                options={resourceGroupOptions}
                value={filters.resourceGroups}
                onChange={(_, value) => handleFilterChange('resourceGroups', value)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Resource Group"
                    placeholder="Select resource groups..."
                  />
                )}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip
                      variant="outlined"
                      label={option}
                      {...getTagProps({ index })}
                      key={option}
                    />
                  ))
                }
              />
            </Grid>

            {showAdvanced && (
              <Grid item xs={12}>
                <Button
                  startIcon={advancedExpanded ? <ExpandLessOutlined /> : <ExpandMoreOutlined />}
                  onClick={() => setAdvancedExpanded(!advancedExpanded)}
                  variant="text"
                  color="primary"
                >
                  Advanced Filters
                </Button>
                
                <Collapse in={advancedExpanded}>
                  <Box sx={{ mt: 2 }}>
                    <Grid container spacing={3}>
                      {/* Location */}
                      <Grid item xs={12} md={6}>
                        <Autocomplete
                          multiple
                          options={locationOptions}
                          value={filters.locations}
                          onChange={(_, value) => handleFilterChange('locations', value)}
                          renderInput={(params) => (
                            <TextField
                              {...params}
                              label="Location"
                              placeholder="Select locations..."
                            />
                          )}
                          renderTags={(value, getTagProps) =>
                            value.map((option, index) => (
                              <Chip
                                variant="outlined"
                                label={option}
                                {...getTagProps({ index })}
                                key={option}
                              />
                            ))
                          }
                        />
                      </Grid>

                      {/* Management Groups */}
                      <Grid item xs={12} md={6}>
                        <Autocomplete
                          multiple
                          options={['Root Management Group', 'Production', 'Development', 'Shared Services']}
                          value={filters.managementGroups}
                          onChange={(_, value) => handleFilterChange('managementGroups', value)}
                          renderInput={(params) => (
                            <TextField
                              {...params}
                              label="Management Group"
                              placeholder="Select management groups..."
                            />
                          )}
                        />
                      </Grid>
                    </Grid>
                  </Box>
                </Collapse>
              </Grid>
            )}
          </Grid>

          {/* Filter Summary */}
          {isFilterActive && (
            <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Active Filters:
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
                {filters.resourceName && (
                  <Chip
                    label={`Name: "${filters.resourceName}"`}
                    onDelete={() => handleFilterChange('resourceName', '')}
                    size="small"
                  />
                )}
                {filters.subscriptions.map(sub => (
                  <Chip
                    key={sub}
                    label={`Sub: ${sub.substring(0, 8)}...`}
                    onDelete={() => handleFilterChange('subscriptions', 
                      filters.subscriptions.filter(s => s !== sub))}
                    size="small"
                  />
                ))}
                {filters.resourceGroups.map(rg => (
                  <Chip
                    key={rg}
                    label={`RG: ${rg}`}
                    onDelete={() => handleFilterChange('resourceGroups', 
                      filters.resourceGroups.filter(r => r !== rg))}
                    size="small"
                  />
                ))}
                {filters.resourceTypes.map(type => (
                  <Chip
                    key={type}
                    label={`Type: ${type.split('/').pop()}`}
                    onDelete={() => handleFilterChange('resourceTypes', 
                      filters.resourceTypes.filter(t => t !== type))}
                    size="small"
                  />
                ))}
                {filters.locations.map(loc => (
                  <Chip
                    key={loc}
                    label={`Location: ${loc}`}
                    onDelete={() => handleFilterChange('locations', 
                      filters.locations.filter(l => l !== loc))}
                    size="small"
                  />
                ))}
              </Stack>
            </Box>
          )}
        </Box>
      </Collapse>
    </Paper>
  )
}

export default GlobalFilterPanel