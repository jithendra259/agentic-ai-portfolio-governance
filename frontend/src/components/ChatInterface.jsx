import React, { useMemo, useState, useEffect, useRef, forwardRef } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import { ChatBox, ChatComposerAttachButton } from '@mui/x-chat';
import { Bot, User, Cpu, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import InlineChart from './InlineChart';

const BACKEND_BASE = 'http://127.0.0.1:8000';
const PLOT_TOKEN = '__PLOTSPEC__:';

// ---------------------------------------------------------------------------
// Markdown image helper — rewrite /outputs/... to full backend URL
// ---------------------------------------------------------------------------
function resolveImageSrc(src) {
  if (!src) return src;
  if (src.startsWith('/outputs/') || src.startsWith('outputs/')) {
    return `${BACKEND_BASE}/${src.replace(/^\//, '')}`;
  }
  return src;
}

// ---------------------------------------------------------------------------
// Preprocess LaTeX delimiters before ReactMarkdown parsing
// remark-math expects $$ and $ but LLMs often output \[ and \(
// ---------------------------------------------------------------------------
function preprocessLaTeX(text) {
  if (!text) return text;
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$')
    .replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');
}

// ---------------------------------------------------------------------------
// renderMarkdown — used as the renderText slot for every chat bubble
//
// Splits the raw text on __PLOTSPEC__:<b64> tokens so charts appear inline
// inside the bubble alongside normal text/markdown.
// ---------------------------------------------------------------------------
function renderMarkdown(rawText) {
  const text = preprocessLaTeX(rawText);

  // Fast path: no embedded chart token
  if (!text.includes(PLOT_TOKEN)) {
    return <MarkdownBlock text={text} />;
  }

  // Split on token boundaries: ["prose...", "__PLOTSPEC__:uuid", "more prose"]
  const parts = text.split(new RegExp(`(${PLOT_TOKEN}[A-Za-z0-9\\-]+)`));
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith(PLOT_TOKEN)) {
          const plotId = part.slice(PLOT_TOKEN.length);
          return <InlineChart key={i} plotId={plotId} />;
        }
        if (part.trim()) return <MarkdownBlock key={i} text={part} />;
        return null;
      })}
    </>
  );
}

const markdownComponents = {
  img: ({ src, alt, ...rest }) => (
    <img
      src={resolveImageSrc(src)}
      alt={alt || 'chart'}
      {...rest}
      style={{ maxWidth: '100%', borderRadius: '8px', marginTop: '8px', display: 'block' }}
    />
  ),
  p: ({ children }) => <p style={{ margin: '4px 0', lineHeight: 1.6 }}>{children}</p>,
  code: ({ children }) => (
    <code style={{ background: '#2A2A2A', color: '#ECECEC', padding: '2px 6px', borderRadius: 4, fontSize: '0.85em' }}>
      {children}
    </code>
  ),
};

function MarkdownBlock({ text }) {
  return (
    <ReactMarkdown 
      components={markdownComponents}
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
    >
      {text}
    </ReactMarkdown>
  );
}

// ---------------------------------------------------------------------------
// Avatar helpers
// ---------------------------------------------------------------------------
function createReactAvatarUrl(IconComponent, background, foreground = '#ffffff') {
  const iconHtml = renderToStaticMarkup(<IconComponent color={foreground} size={48} />);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
    <rect width="96" height="96" rx="24" fill="${background}"/>
    <svg x="24" y="24" width="48" height="48">${iconHtml}</svg>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const botUser = {
  id: 'assistant',
  displayName: 'Portfolio AI',
  avatarUrl: createReactAvatarUrl(Bot, '#404040', '#ECECEC'),
  isOnline: true,
};

const youUser = {
  id: 'user',
  displayName: 'You',
  avatarUrl: createReactAvatarUrl(User, '#2A2A2A', '#ECECEC'),
  isOnline: true,
};

// ---------------------------------------------------------------------------
// NDJSON stream parser
// ---------------------------------------------------------------------------
async function parseNDJSONStream(response, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  return new ReadableStream({
    async pull(controller) {
      if (signal?.aborted) { controller.error(new Error('Aborted')); return; }
      const { done, value } = await reader.read();
      if (done) {
        if (buffer.trim()) {
          try { controller.enqueue(JSON.parse(buffer)); } catch { }
        }
        controller.close();
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          try { controller.enqueue(JSON.parse(trimmed)); } catch (e) {
            console.error('Failed to parse NDJSON chunk:', trimmed);
          }
        }
      }
    },
    cancel() { reader.cancel(); },
  });
}

// ---------------------------------------------------------------------------
// Custom Composer Attach Button with inline model selector
// ---------------------------------------------------------------------------
const CustomAttachButtonWithModelSelector = forwardRef(({
  selectedModel,
  setSelectedModel,
  availableModels,
  loadingModels,
  ...otherProps
}, ref) => {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 1 }}>
      <ChatComposerAttachButton ref={ref} {...otherProps} />
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, pl: 0.5 }}>
        <Cpu size={14} color="rgba(255, 255, 255, 0.45)" style={{ transition: 'all 0.3s ease' }} />
        <FormControl size="small">
          <Select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            displayEmpty
            IconComponent={ChevronDown}
            sx={{
              height: 30,
              minWidth: 150,
              fontSize: '0.78rem',
              fontWeight: 500,
              color: 'rgba(255, 255, 255, 0.85)',
              fontFamily: 'system-ui, -apple-system, sans-serif',
              backgroundColor: 'rgba(255, 255, 255, 0.03)',
              borderRadius: '6px',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              '&:hover': {
                borderColor: 'rgba(255, 255, 255, 0.2)',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
              },
              '&.Mui-focused': {
                borderColor: 'rgba(255, 255, 255, 0.4)',
                backgroundColor: 'rgba(255, 255, 255, 0.06)',
                boxShadow: '0 0 12px rgba(255, 255, 255, 0.03)',
              },
              '& .MuiOutlinedInput-notchedOutline': {
                border: 'none',
              },
              '& .MuiSelect-select': {
                paddingLeft: '8px',
                paddingRight: '28px',
                paddingTop: '2px',
                paddingBottom: '2px',
                display: 'flex',
                alignItems: 'center',
              },
              '& .MuiSelect-icon': {
                color: 'rgba(255, 255, 255, 0.4)',
                right: '6px',
                width: '14px',
                height: '14px',
                transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              },
              '&.Mui-focused .MuiSelect-icon': {
                transform: 'rotate(180deg)',
                color: 'rgba(255, 255, 255, 0.8)',
              },
            }}
            MenuProps={{
              TransitionProps: { timeout: 150 },
              PaperProps: {
                sx: {
                  backgroundColor: 'rgba(18, 18, 18, 0.95)',
                  backdropFilter: 'blur(16px)',
                  color: '#ECECEC',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '8px',
                  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
                  marginTop: '6px',
                  '& .MuiMenuItem-root': {
                    fontSize: '0.78rem',
                    fontWeight: 500,
                    padding: '6px 12px',
                    margin: '2px 4px',
                    borderRadius: '4px',
                    transition: 'all 0.15s ease',
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.06)',
                      color: '#FFFFFF',
                    },
                    '&.Mui-selected': {
                      backgroundColor: 'rgba(0, 229, 255, 0.12)',
                      color: '#00E5FF',
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'rgba(0, 229, 255, 0.18)',
                      },
                    },
                  },
                },
              },
            }}
          >
            {loadingModels ? (
              <MenuItem disabled value="">
                Loading...
              </MenuItem>
            ) : availableModels.length === 0 ? (
              <MenuItem disabled value="">
                No models available
              </MenuItem>
            ) : (
              availableModels.map((model) => (
                <MenuItem key={model} value={model}>
                  {model}
                </MenuItem>
              ))
            )}
          </Select>
        </FormControl>
      </Box>
    </Box>
  );
});

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ChatInterface() {
  const sessionId = 'demo-react-session';

  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const selectedModelRef = useRef('');

  useEffect(() => {
    selectedModelRef.current = selectedModel;
  }, [selectedModel]);

  useEffect(() => {
    let active = true;
    fetch(`${BACKEND_BASE}/health`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch health');
        return res.json();
      })
      .then((data) => {
        if (!active) return;
        const models = data?.models?.available || [];
        const primary = data?.models?.primary || '';
        setAvailableModels(models);
        if (primary && models.includes(primary)) {
          setSelectedModel(primary);
        } else if (models.length > 0) {
          setSelectedModel(models[0]);
        }
        setLoadingModels(false);
      })
      .catch((err) => {
        console.error('Failed to fetch Ollama models from backend:', err);
        if (active) {
          setLoadingModels(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const adapter = useMemo(() => ({
    async sendMessage({ message, signal }) {
      const textPart = message.parts?.find(p => p.type === 'text');
      const userText = textPart ? textPart.text : (typeof message === 'string' ? message : '');

      const response = await fetch(`${BACKEND_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          session_id: sessionId, 
          user_message: userText,
          model: selectedModelRef.current || null
        }),
        signal,
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const ndjsonStream = await parseNDJSONStream(response, signal);

      // pendingTextId tracks which text part is currently open so we can
      // inject chart tokens into the same message after the main text ends.
      let pendingTextId = null;

      const transformStream = new TransformStream({
        transform(chunk, controller) {
          if (chunk.type === 'text-start') {
            pendingTextId = chunk.id;
            controller.enqueue(chunk);
          } else if (chunk.type === 'text-end') {
            pendingTextId = null;
            controller.enqueue(chunk);
          } else if (chunk.type === 'data-plot') {
            const plotId = chunk.plotId;

            if (plotId) {
              const token = `\n${PLOT_TOKEN}${plotId}`;
              const randomId = Math.random().toString(36).substring(2, 11);
              const chartTextId = `chart-${randomId}`;

              controller.enqueue({ type: 'text-start', id: chartTextId });
              controller.enqueue({ type: 'text-delta', id: chartTextId, delta: token });
              controller.enqueue({ type: 'text-end', id: chartTextId });
            }
            // Don't forward the raw data-plot chunk — it's not a standard MUI x-chat type
          } else {
            controller.enqueue(chunk);
          }
        },
      });

      return ndjsonStream.pipeThrough(transformStream);
    },
  }), [sessionId]);

  const initialMessages = useMemo(() => [
    {
      id: 'msg-welcome-1',
      senderId: 'assistant',
      createdAt: new Date(),
      parts: [
        {
          type: 'text',
          text: 'Hello! I am your Portfolio Assistant. How can I help you analyze your portfolio today?'
        }
      ]
    }
  ], []);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
      }}
    >
      <ChatBox
        adapter={adapter}
        initialConversations={[{
          id: sessionId,
          title: 'Portfolio Assistant',
          participants: [youUser, botUser],
        }]}
        initialActiveConversationId={sessionId}
        initialMessages={initialMessages}
        slots={{
          composerAttachButton: CustomAttachButtonWithModelSelector,
        }}
        slotProps={{
          messageContent: {
            partProps: {
              text: { renderText: renderMarkdown },
            },
          },
          composerInput: {
            sx: {
              color: '#ECECEC',
              backgroundColor: '#1A1A1A',
              '& .MuiOutlinedInput-root': {
                color: '#ECECEC',
                '& fieldset': {
                  borderColor: '#404040',
                },
                '&:hover fieldset': {
                  borderColor: '#666666',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#FFFFFF',
                },
              },
            }
          },
          composerAttachButton: {
            selectedModel,
            setSelectedModel,
            availableModels,
            loadingModels,
          }
        }}
        suggestions={[
          'Plot AAPL and MSFT prices from 2020 to 2024',
          'Show me the optimal portfolio allocation',
          'Analyse systemic risk for my portfolio',
          'What is the governance score for TSLA?',
        ]}
        suggestionsAutoSubmit={false}
        sx={{
          flex: 1,
          height: '100%',
          border: '1px solid #404040',
          borderRadius: 2,
          backgroundColor: '#0D0D0D',
          color: '#ECECEC',
        }}
      />
    </Box>
  );
}