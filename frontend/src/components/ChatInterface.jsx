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
import remarkGfm from 'remark-gfm';
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
  table: ({ children }) => (
    <div className="markdown-table-container">
      <table className="markdown-table">
        {children}
      </table>
    </div>
  ),
};

function MarkdownBlock({ text }) {
  return (
    <ReactMarkdown 
      components={markdownComponents}
      remarkPlugins={[remarkMath, remarkGfm]}
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
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, ml: 1 }}>
      <ChatComposerAttachButton ref={ref} {...otherProps} />
      <FormControl size="small">
        <Select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          displayEmpty
          IconComponent={ChevronDown}
          sx={{
            height: 34,
            minWidth: 130,
            fontSize: '0.88rem !important',
            fontWeight: '600 !important',
            color: '#B4B4B4 !important',
            fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important',
            letterSpacing: '-0.01em !important',
            backgroundColor: 'transparent !important', // Ensure no global MuiInputBase grey box
            borderRadius: '8px',
            transition: 'all 0.15s ease',
            border: 'none !important',
            boxShadow: 'none !important',
            '&:hover': {
              color: '#FFFFFF !important',
              backgroundColor: 'rgba(255, 255, 255, 0.06) !important',
            },
            '&.Mui-focused': {
              color: '#FFFFFF !important',
              backgroundColor: 'rgba(255, 255, 255, 0.06) !important',
            },
            '& .MuiOutlinedInput-notchedOutline': {
              border: 'none !important', // Strictly no border outline
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              border: 'none !important',
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              border: 'none !important',
            },
            '& .MuiSelect-select': {
              paddingLeft: '8px !important',
              paddingRight: '28px !important',
              paddingTop: '4px !important',
              paddingBottom: '4px !important',
              display: 'flex !important',
              alignItems: 'center !important',
              backgroundColor: 'transparent !important',
            },
            '& .MuiSelect-icon': {
              color: '#B4B4B4 !important',
              right: '8px !important',
              width: '15px !important',
              height: '15px !important',
              transition: 'transform 0.15s ease, color 0.15s ease !important',
            },
            '&:hover .MuiSelect-icon': {
              color: '#FFFFFF !important',
            },
            '&.Mui-focused .MuiSelect-icon': {
              transform: 'rotate(180deg) !important',
              color: '#FFFFFF !important',
            },
          }}
          MenuProps={{
            TransitionProps: { timeout: 120 },
            PaperProps: {
              sx: {
                backgroundColor: '#2F2F2F !important', // ChatGPT pop-up color
                color: '#FFFFFF !important',
                border: '1px solid #3F3F3F !important',
                borderRadius: '10px !important',
                boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5) !important',
                marginTop: '6px',
                transformOrigin: 'top center',
                '& .MuiList-root': {
                  padding: '4px 0 !important',
                },
                '& .MuiMenuItem-root': {
                  fontSize: '0.92rem !important',
                  fontWeight: '500 !important',
                  padding: '12px 18px !important', // Generous clean padding
                  transition: 'background-color 0.1s ease, color 0.1s ease !important',
                  fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important',
                  letterSpacing: '-0.015em !important',
                  color: '#ECECEC !important',
                  borderRadius: '0 !important', // Edge-to-edge classic list item highlight
                  '&:hover': {
                    backgroundColor: '#3E3E3E !important', // Dark gray highlight matching screenshot
                    color: '#FFFFFF !important',
                  },
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08) !important',
                    color: '#FFFFFF !important',
                    fontWeight: '600 !important',
                    '&:hover': {
                      backgroundColor: '#3E3E3E !important',
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