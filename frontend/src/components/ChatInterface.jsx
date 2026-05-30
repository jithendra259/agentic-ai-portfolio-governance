import React, { useState, useRef, useEffect } from 'react';
import { Box, TextField, IconButton, Typography, Paper, CircularProgress } from '@mui/material';
import { Send, User, Bot } from 'lucide-react';
import InteractivePlot from './InteractivePlot';

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Use a fixed session ID for the demo
  const sessionId = "demo-react-session";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_message: userMsg,
        })
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      setMessages(prev => [...prev, { role: 'assistant', content: '', status: 'Thinking...', isStreaming: true }]);

      let assistantContent = '';
      let statusText = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(Boolean);
        
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            if (data.type === 'token') {
              assistantContent += data.content;
            } else if (data.type === 'status') {
              statusText = data.content;
            } else if (data.type === 'final') {
              assistantContent = data.content;
            }
            
            setMessages(prev => {
              const newMsgs = [...prev];
              newMsgs[newMsgs.length - 1] = {
                role: 'assistant',
                content: assistantContent,
                status: statusText,
                isStreaming: true
              };
              return newMsgs;
            });
            
          } catch (e) {
            console.error('Error parsing SSE:', e);
          }
        }
      }

      // Stream finished
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], isStreaming: false, status: '' };
        return newMsgs;
      });

      // Check if the response contains the special plot message
      if (assistantContent.includes('Plot data successfully loaded!')) {
        // Fetch the plot data!
        const plotRes = await fetch(`http://127.0.0.1:8000/plot_data/${sessionId}`);
        if (plotRes.ok) {
          const plotData = await plotRes.json();
          if (plotData && plotData.data) {
             setMessages(prev => [
               ...prev,
               { role: 'assistant', type: 'plot', plotData: plotData.data, plotTitle: plotData.title }
             ]);
          }
        }
      }

    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev, 
        { role: 'assistant', content: 'An error occurred while connecting to the server.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '1200px', mx: 'auto', p: 2 }}>
      <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 2, color: 'primary.main', textAlign: 'center' }}>
        Agentic Portfolio Governance
      </Typography>
      
      <Paper elevation={3} sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', bgcolor: 'background.paper' }}>
        {/* Chat History */}
        <Box sx={{ flex: 1, overflowY: 'auto', p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
          {messages.length === 0 && (
            <Box sx={{ m: 'auto', textAlign: 'center', color: 'text.secondary' }}>
              <Bot size={48} opacity={0.5} style={{ marginBottom: 16 }} />
              <Typography variant="h6">Welcome to the Advisory Backend.</Typography>
              <Typography variant="body2">Ask me to plot historical returns for a universe.</Typography>
            </Box>
          )}

          {messages.map((msg, i) => (
            <Box 
              key={i} 
              sx={{ 
                display: 'flex', 
                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                gap: 2,
                alignItems: 'flex-start'
              }}
            >
              <Box sx={{ 
                bgcolor: msg.role === 'user' ? 'primary.main' : 'background.default',
                p: 1, 
                borderRadius: '50%',
                display: 'flex'
              }}>
                {msg.role === 'user' ? <User size={20} color="white" /> : <Bot size={20} color="#9ca3af" />}
              </Box>
              
              {msg.type === 'plot' ? (
                <Box sx={{ width: '80%' }}>
                  <InteractivePlot data={msg.plotData} title={msg.plotTitle} />
                </Box>
              ) : (
                <Paper 
                  elevation={1}
                  sx={{ 
                    p: 2, 
                    maxWidth: '80%', 
                    bgcolor: msg.role === 'user' ? 'primary.dark' : 'background.default',
                    color: msg.role === 'user' ? 'white' : 'text.primary',
                    borderRadius: 2,
                    borderTopRightRadius: msg.role === 'user' ? 0 : 8,
                    borderTopLeftRadius: msg.role === 'user' ? 8 : 0,
                    whiteSpace: 'pre-wrap'
                  }}
                >
                  <Typography variant="body1">{msg.content}</Typography>
                  {msg.isStreaming && msg.status && (
                    <Typography variant="caption" sx={{ display: 'block', mt: 1, color: '#34d399', fontStyle: 'italic' }}>
                      {msg.status}
                    </Typography>
                  )}
                </Paper>
              )}
            </Box>
          ))}
          <div ref={messagesEndRef} />
        </Box>

        {/* Input Area */}
        <Box sx={{ p: 2, bgcolor: 'background.default', borderTop: '1px solid', borderColor: 'divider' }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Ask about historical performance or plot returns..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={isLoading}
            multiline
            maxRows={4}
            InputProps={{
              endAdornment: (
                <IconButton 
                  color="primary" 
                  onClick={handleSend} 
                  disabled={!input.trim() || isLoading}
                >
                  {isLoading ? <CircularProgress size={24} /> : <Send size={24} />}
                </IconButton>
              )
            }}
            sx={{ bgcolor: 'background.paper', borderRadius: 1 }}
          />
        </Box>
      </Paper>
    </Box>
  );
}
