import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  List,
  ListItem,
  Avatar,
  Chip,
  LinearProgress,
  Alert,
  Divider,
  Card,
  CardContent,
  IconButton,
  Menu,
  MenuItem,
  Tooltip,
  Badge
} from '@mui/material'
import {
  ChatOutlined,
  SendOutlined,
  SmartToyOutlined,
  PersonOutlined,
  MoreVertOutlined,
  PolicyOutlined,
  SecurityOutlined,
  MonetizationOnOutlined,
  AnalyticsOutlined,
  DeleteOutlined,
  HistoryOutlined,
  AutoAwesomeOutlined
} from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'
import { apiClient } from '../../services/apiClient'

interface ConversationMessage {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  intent?: string
  entities?: Record<string, string[]>
  confidence?: number
  suggestions?: string[]
  metadata?: Record<string, any>
}

interface ConversationSession {
  sessionId: string
  messages: ConversationMessage[]
  isActive: boolean
  lastActivity: Date
}

const ConversationPage = () => {
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>()
  const [error, setError] = useState<string | null>(null)
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null)
  const [suggestions] = useState([
    "What are the current security policies for virtual machines?",
    "Check our GDPR compliance status",
    "How can we reduce Azure costs for compute resources?",
    "Show me policy violations from last week",
    "Analyze network security group configurations"
  ])
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Initialize session
  useEffect(() => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    setSessionId(newSessionId)
    
    // Add welcome message
    const welcomeMessage: ConversationMessage = {
      id: 'welcome',
      type: 'assistant',
      content: '👋 Hello! I\'m your AI-powered governance assistant. I can help you with Azure policies, compliance checks, cost optimization, and security analysis. What would you like to know?',
      timestamp: new Date(),
      suggestions: suggestions.slice(0, 3)
    }
    setMessages([welcomeMessage])
  }, [])

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !sessionId) return

    const userMessage: ConversationMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)
    setError(null)

    try {
      const response = await apiClient.post('/api/v1/conversation/governance', {
        user_input: inputMessage.trim(),
        session_id: sessionId,
        user_id: 'current_user' // Replace with actual user ID
      })

      if (response.data.response) {
        const assistantMessage: ConversationMessage = {
          id: `assistant_${Date.now()}`,
          type: 'assistant',
          content: response.data.response,
          timestamp: new Date(),
          intent: response.data.intent,
          entities: response.data.entities,
          confidence: response.data.confidence,
          suggestions: response.data.suggestions,
          metadata: response.data.azure_context
        }

        setMessages(prev => [...prev, assistantMessage])
      } else {
        throw new Error('Failed to process conversation')
      }
    } catch (error: any) {
      console.error('Conversation error:', error)
      setError('Sorry, I encountered an error processing your request. Please try again.')
      
      const errorMessage: ConversationMessage = {
        id: `error_${Date.now()}`,
        type: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again or rephrase your question.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSendMessage()
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInputMessage(suggestion)
    inputRef.current?.focus()
  }

  const clearConversation = () => {
    setMessages(messages.slice(0, 1)) // Keep welcome message
    setMenuAnchor(null)
  }

  const getIntentIcon = (intent?: string) => {
    switch (intent) {
      case 'policy_query':
        return <PolicyOutlined />
      case 'security_analysis':
        return <SecurityOutlined />
      case 'cost_optimization':
        return <MonetizationOnOutlined />
      case 'compliance_check':
        return <AnalyticsOutlined />
      default:
        return <AutoAwesomeOutlined />
    }
  }

  const getIntentColor = (intent?: string) => {
    switch (intent) {
      case 'policy_query':
        return 'primary'
      case 'security_analysis':
        return 'error'
      case 'cost_optimization':
        return 'success'
      case 'compliance_check':
        return 'warning'
      default:
        return 'default'
    }
  }

  return (
    <>
      <Helmet>
        <title>AI Governance Assistant - PolicyCortex</title>
        <meta name="description" content="Conversational AI assistant for Azure governance, policies, and compliance" />
      </Helmet>

      <Box sx={{ height: 'calc(100vh - 120px)', display: 'flex', flexDirection: 'column', p: 2 }}>
        {/* Header */}
        <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Badge color="success" variant="dot">
                <SmartToyOutlined color="primary" />
              </Badge>
              <Typography variant="h5" component="h1">
                AI Governance Assistant
              </Typography>
              <Chip
                label="Patent 3: Conversational Intelligence"
                size="small"
                color="secondary"
                variant="outlined"
              />
            </Box>
            <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
              <MoreVertOutlined />
            </IconButton>
          </Box>
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Ask me about Azure policies, compliance, security, cost optimization, and governance best practices.
          </Typography>
        </Paper>

        {/* Menu */}
        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={clearConversation}>
            <DeleteOutlined sx={{ mr: 1 }} />
            Clear Conversation
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            <HistoryOutlined sx={{ mr: 1 }} />
            View History
          </MenuItem>
        </Menu>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Messages Area */}
        <Paper elevation={1} sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
            <List sx={{ py: 0 }}>
              {messages.map((message, index) => (
                <React.Fragment key={message.id}>
                  <ListItem sx={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch', py: 2 }}>
                    <Box sx={{ display: 'flex', gap: 2, width: '100%' }}>
                      <Avatar sx={{ bgcolor: message.type === 'user' ? 'primary.main' : 'secondary.main' }}>
                        {message.type === 'user' ? <PersonOutlined /> : <SmartToyOutlined />}
                      </Avatar>
                      
                      <Box sx={{ flex: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Typography variant="subtitle2" color="text.secondary">
                            {message.type === 'user' ? 'You' : 'AI Assistant'}
                          </Typography>
                          <Typography variant="caption" color="text.disabled">
                            {message.timestamp.toLocaleTimeString()}
                          </Typography>
                          {message.intent && (
                            <Chip
                              icon={getIntentIcon(message.intent)}
                              label={message.intent.replace('_', ' ')}
                              size="small"
                              color={getIntentColor(message.intent) as any}
                              variant="outlined"
                            />
                          )}
                          {message.confidence && (
                            <Tooltip title={`Confidence: ${Math.round(message.confidence * 100)}%`}>
                              <Chip
                                label={`${Math.round(message.confidence * 100)}%`}
                                size="small"
                                color={message.confidence > 0.8 ? 'success' : message.confidence > 0.6 ? 'warning' : 'error'}
                                variant="outlined"
                              />
                            </Tooltip>
                          )}
                        </Box>
                        
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', mb: 1 }}>
                          {message.content}
                        </Typography>

                        {/* Entities */}
                        {message.entities && Object.keys(message.entities).length > 0 && (
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                              Detected Entities:
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {Object.entries(message.entities).map(([entityType, values]) =>
                                values.map((value, idx) => (
                                  <Chip
                                    key={`${entityType}_${idx}`}
                                    label={`${entityType}: ${value}`}
                                    size="small"
                                    variant="outlined"
                                  />
                                ))
                              )}
                            </Box>
                          </Box>
                        )}

                        {/* Suggestions */}
                        {message.suggestions && message.suggestions.length > 0 && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                              Suggested questions:
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                              {message.suggestions.map((suggestion, idx) => (
                                <Button
                                  key={idx}
                                  size="small"
                                  variant="outlined"
                                  onClick={() => handleSuggestionClick(suggestion)}
                                  sx={{ textTransform: 'none' }}
                                >
                                  {suggestion}
                                </Button>
                              ))}
                            </Box>
                          </Box>
                        )}

                        {/* Metadata (for debugging) */}
                        {message.metadata && process.env.NODE_ENV === 'development' && (
                          <Card variant="outlined" sx={{ mt: 1, bgcolor: 'grey.50' }}>
                            <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                              <Typography variant="caption" color="text.secondary">
                                Debug Info: {JSON.stringify(message.metadata, null, 2)}
                              </Typography>
                            </CardContent>
                          </Card>
                        )}
                      </Box>
                    </Box>
                  </ListItem>
                  {index < messages.length - 1 && <Divider variant="middle" />}
                </React.Fragment>
              ))}
            </List>
            
            {/* Loading indicator */}
            {isLoading && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2 }}>
                <Avatar sx={{ bgcolor: 'secondary.main' }}>
                  <SmartToyOutlined />
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    AI Assistant is thinking...
                  </Typography>
                  <LinearProgress />
                </Box>
              </Box>
            )}
            
            <div ref={messagesEndRef} />
          </Box>

          {/* Input Area */}
          <Divider />
          <Box sx={{ p: 2 }}>
            {/* Quick suggestions when no conversation */}
            {messages.length === 1 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                  Try asking:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {suggestions.map((suggestion, idx) => (
                    <Button
                      key={idx}
                      size="small"
                      variant="outlined"
                      onClick={() => handleSuggestionClick(suggestion)}
                      sx={{ textTransform: 'none' }}
                    >
                      {suggestion}
                    </Button>
                  ))}
                </Box>
              </Box>
            )}

            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                ref={inputRef}
                fullWidth
                multiline
                maxRows={4}
                placeholder="Ask about Azure governance, policies, compliance, or security..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                variant="outlined"
                size="small"
              />
              <Button
                variant="contained"
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                sx={{ minWidth: 48, px: 2 }}
              >
                <SendOutlined />
              </Button>
            </Box>
          </Box>
        </Paper>
      </Box>
    </>
  )
}

export default ConversationPage