import { Box, Typography, Paper } from '@mui/material'
import { ChatOutlined } from '@mui/icons-material'
import { Helmet } from 'react-helmet-async'

const ConversationPage = () => {
  return (
    <>
      <Helmet>
        <title>AI Assistant - PolicyCortex</title>
        <meta name="description" content="Chat with the AI assistant for Azure governance insights" />
      </Helmet>

      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ChatOutlined />
          AI Assistant
        </Typography>
        
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Conversation Interface
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This page will contain the AI conversation interface with WebSocket support.
          </Typography>
        </Paper>
      </Box>
    </>
  )
}

export default ConversationPage