import { useState } from 'react'
import {
  Box,
  TextField,
  InputAdornment,
  IconButton,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
  Divider,
  useTheme,
  alpha,
} from '@mui/material'
import { SearchOutlined, ClearOutlined } from '@mui/icons-material'
import { useDebounce } from 'use-debounce'

interface SearchResult {
  id: string
  title: string
  description: string
  type: string
  icon: React.ReactNode
  url: string
}

interface SearchBarProps {
  placeholder?: string
  onSearch?: (query: string) => void
  onResultClick?: (result: SearchResult) => void
}

export const SearchBar = ({
  placeholder = 'Search policies, resources, costs...',
  onSearch,
  onResultClick,
}: SearchBarProps) => {
  const theme = useTheme()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [showResults, setShowResults] = useState(false)
  const [debouncedQuery] = useDebounce(query, 300)

  const handleSearch = (value: string) => {
    setQuery(value)
    
    if (value.length > 2) {
      // TODO: Implement actual search logic
      const mockResults: SearchResult[] = [
        {
          id: '1',
          title: 'Azure Policy: Require SSL',
          description: 'Policy to enforce SSL on all web apps',
          type: 'Policy',
          icon: <SearchOutlined />,
          url: '/policies/1',
        },
        {
          id: '2',
          title: 'VM-WebServer-001',
          description: 'Virtual machine in production',
          type: 'Resource',
          icon: <SearchOutlined />,
          url: '/resources/2',
        },
      ]
      
      setResults(mockResults)
      setShowResults(true)
    } else {
      setResults([])
      setShowResults(false)
    }
    
    onSearch?.(value)
  }

  const handleClear = () => {
    setQuery('')
    setResults([])
    setShowResults(false)
  }

  const handleResultClick = (result: SearchResult) => {
    setShowResults(false)
    onResultClick?.(result)
  }

  return (
    <Box sx={{ position: 'relative' }}>
      <TextField
        fullWidth
        variant="outlined"
        placeholder={placeholder}
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        onFocus={() => setShowResults(results.length > 0)}
        onBlur={() => setTimeout(() => setShowResults(false), 200)}
        size="small"
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchOutlined sx={{ color: 'text.secondary' }} />
            </InputAdornment>
          ),
          endAdornment: query && (
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={handleClear}
                sx={{ color: 'text.secondary' }}
              >
                <ClearOutlined />
              </IconButton>
            </InputAdornment>
          ),
          sx: {
            backgroundColor: alpha(theme.palette.background.paper, 0.8),
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: alpha(theme.palette.divider, 0.5),
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: theme.palette.primary.main,
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              borderColor: theme.palette.primary.main,
            },
          },
        }}
      />

      {showResults && results.length > 0 && (
        <Paper
          elevation={8}
          sx={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            zIndex: 1000,
            mt: 1,
            maxHeight: 400,
            overflow: 'auto',
          }}
        >
          <List disablePadding>
            <ListItem>
              <Typography variant="body2" color="text.secondary">
                Search results for "{query}"
              </Typography>
            </ListItem>
            <Divider />
            
            {results.map((result) => (
              <ListItem
                key={result.id}
                button
                onClick={() => handleResultClick(result)}
                sx={{
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.05),
                  },
                }}
              >
                <ListItemIcon>{result.icon}</ListItemIcon>
                <ListItemText
                  primary={result.title}
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {result.description}
                      </Typography>
                      <Typography variant="caption" color="primary">
                        {result.type}
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  )
}