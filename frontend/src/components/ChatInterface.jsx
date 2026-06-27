import { useMemo, useState, useEffect, useRef, useCallback, forwardRef } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { UserButton } from '@clerk/react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Button from '@mui/material/Button';
import {
  Bot,
  User,
  ChevronDown,
  SquarePen,
  BarChart2,
  PanelLeft,
  Plus,
  Share,
  MoreHorizontal,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Upload,
  Mic,
  AudioLines,
  Search,
  Sparkles,
  Trash2,
  Square,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import InlineChart from './InlineChart';
import PlotFixtureGallery from './PlotFixtureGallery';
import { BACKEND_BASE } from '../config/api';
import { useAuth } from '../context/AuthContext';
import {
  readCachedMessages,
  readCachedSessions,
  writeCachedMessages,
  writeCachedSessions,
} from '../utils/chatHistoryCache';

const PLOT_TOKEN = '__PLOTSPEC__:';
const SESSION_STORAGE_KEY = 'portfolio-ai-chat-session-id';
const SESSION_INDEX_STORAGE_KEY = 'portfolio-ai-chat-session-ids';
const WELCOME_MESSAGE_ID = 'msg-welcome-1';

function createSessionId() {
  const randomPart = window.crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
  return `portfolio-chat-${randomPart}`;
}

function readStoredSessionIds(storageKey = SESSION_INDEX_STORAGE_KEY) {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) || '[]');
    return Array.isArray(parsed)
      ? parsed.map((item) => String(item || '').trim()).filter(Boolean)
      : [];
  } catch (_) {
    return [];
  }
}

function writeStoredSessionIds(sessionIds, storageKey = SESSION_INDEX_STORAGE_KEY) {
  const unique = [];
  for (const sessionId of sessionIds || []) {
    const value = String(sessionId || '').trim();
    if (value && !unique.includes(value)) unique.push(value);
  }
  window.localStorage.setItem(storageKey, JSON.stringify(unique.slice(0, 100)));
}

function rememberSessionIds(sessionIds, storageKey = SESSION_INDEX_STORAGE_KEY) {
  writeStoredSessionIds([...readStoredSessionIds(storageKey), ...(sessionIds || [])], storageKey);
}

function rememberSessionId(sessionId, storageKey = SESSION_INDEX_STORAGE_KEY) {
  rememberSessionIds([sessionId], storageKey);
}

function forgetSessionId(sessionId, storageKey = SESSION_INDEX_STORAGE_KEY) {
  writeStoredSessionIds(readStoredSessionIds(storageKey).filter((item) => item !== sessionId), storageKey);
}

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
  const parts = text.split(new RegExp(`(${PLOT_TOKEN}[A-Za-z0-9_-]+)`));
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
  role: 'assistant',
};

const youUser = {
  id: 'user',
  displayName: 'You',
  avatarUrl: createReactAvatarUrl(User, '#2A2A2A', '#ECECEC'),
  isOnline: true,
  role: 'user',
};

function makeConversation(sessionId, messages = [], summary = null) {
  const lastMessage = messages[messages.length - 1];
  const lastText = getMessagePreview(lastMessage);

  return {
    id: sessionId,
    title: summary?.title || 'Portfolio Assistant',
    subtitle: lastText || (summary ? `${summary.message_count} messages` : 'Ask about portfolio governance, charts, and risk'),
    participants: [youUser, botUser],
    readState: 'read',
    unreadCount: 0,
    lastMessageAt: lastMessage?.createdAt || summary?.updated_at || new Date().toISOString(),
  };
}

function upsertConversation(conversations, conversation) {
  const withoutCurrent = conversations.filter((item) => item.id !== conversation.id);
  return [conversation, ...withoutCurrent].sort(
    (a, b) => new Date(b.lastMessageAt || 0) - new Date(a.lastMessageAt || 0),
  );
}

function makeWelcomeMessage(sessionId) {
  return {
    id: WELCOME_MESSAGE_ID,
    conversationId: sessionId,
    role: 'assistant',
    status: 'sent',
    createdAt: new Date().toISOString(),
    author: botUser,
    parts: [
      {
        type: 'text',
        text: 'Hello! I am your Portfolio Assistant. How can I help you analyze your portfolio today?',
        state: 'done',
      },
    ],
  };
}

function toChatMessage(item, sessionId) {
  const role = item.role === 'assistant' ? 'assistant' : 'user';

  return {
    id: `persisted-${item.id}`,
    conversationId: sessionId,
    role,
    status: 'sent',
    createdAt: item.created_at || new Date().toISOString(),
    author: role === 'assistant' ? botUser : youUser,
    metadata: item.metadata || {},
    parts: [
      {
        type: 'text',
        text: item.content || '',
        state: 'done',
      },
    ],
  };
}

function getMessagePreview(message) {
  const text = message?.parts
    ?.filter((part) => part.type === 'text')
    .map((part) => part.text || '')
    .join(' ')
    .replace(new RegExp(`${PLOT_TOKEN}[A-Za-z0-9_-]+`, 'g'), '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!text) return '';
  return text.length > 64 ? `${text.slice(0, 61)}...` : text;
}

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
          try {
            controller.enqueue(JSON.parse(buffer));
          } catch {
            console.error('Failed to parse trailing NDJSON chunk:', buffer);
          }
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
          try {
            controller.enqueue(JSON.parse(trimmed));
          } catch {
            console.error('Failed to parse NDJSON chunk:', trimmed);
          }
        }
      }
    },
    cancel() { reader.cancel(); },
  });
}

async function readApiError(response, fallbackMessage) {
  try {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      const payload = await response.json();
      return payload?.detail || payload?.message || fallbackMessage;
    }
    const text = await response.text();
    return text || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
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
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
      <Tooltip title="Attach">
        <IconButton ref={ref} {...otherProps} className="composer-icon-button" aria-label="Attach file">
          <Plus size={24} />
        </IconButton>
      </Tooltip>
      <FormControl size="small">
        <Select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          displayEmpty
          IconComponent={ChevronDown}
          sx={{
            height: 34,
            minWidth: 122,
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

const NewChatButton = forwardRef(({ onNewChat, ...props }, ref) => (
  <Tooltip title="New chat">
    <IconButton
      ref={ref}
      aria-label="New chat"
      onClick={onNewChat}
      size="small"
      {...props}
      sx={{
        width: 36,
        height: 36,
        color: '#ECECEC',
        borderRadius: '8px',
        '&:hover': {
          backgroundColor: 'rgba(255, 255, 255, 0.08)',
        },
        ...props.sx,
      }}
    >
      <SquarePen size={18} />
    </IconButton>
  </Tooltip>
));

function messageText(message) {
  return message?.parts
    ?.filter((part) => part.type === 'text')
    .map((part) => part.text || '')
    .join('') || '';
}

function patchAssistantMessage(messages, assistantId, updater) {
  return messages.map((message) => {
    if (message.id !== assistantId) return message;
    const text = messageText(message);
    return {
      ...message,
      status: 'streaming',
      parts: [{ type: 'text', text: updater(text), state: 'streaming' }],
    };
  });
}

function ChatMessageActions({ text, onRegenerate }) {
  const copyText = () => {
    navigator.clipboard?.writeText(text || '').catch(() => {});
  };
  return (
    <Box className="message-actions">
      <Tooltip title="Copy response">
        <IconButton size="small" onClick={copyText}><Copy size={18} /></IconButton>
      </Tooltip>
      <Tooltip title="Good response">
        <IconButton size="small"><ThumbsUp size={18} /></IconButton>
      </Tooltip>
      <Tooltip title="Bad response">
        <IconButton size="small"><ThumbsDown size={18} /></IconButton>
      </Tooltip>
      <Tooltip title="Share">
        <IconButton size="small"><Upload size={18} /></IconButton>
      </Tooltip>
      <Tooltip title="Regenerate">
        <IconButton size="small" onClick={onRegenerate}><RotateCcw size={18} /></IconButton>
      </Tooltip>
      <Tooltip title="More">
        <IconButton size="small"><MoreHorizontal size={18} /></IconButton>
      </Tooltip>
    </Box>
  );
}

function ChatMessageRow({ message, onRegenerate }) {
  const role = message.role || (message.senderId === 'assistant' ? 'assistant' : 'user');
  const text = messageText(message);
  const isUser = role === 'user';
  const isStreaming = message.status === 'streaming'
    || message.parts?.some((part) => part.state === 'streaming');
  const isWelcomeMessage = message.id === WELCOME_MESSAGE_ID;
  const showActions = !isUser && !isWelcomeMessage && !isStreaming && text.trim().length > 0;
  return (
    <Box className={`message-row ${isUser ? 'message-row-user' : 'message-row-assistant'}`}>
      <Box className="message-content">
        {!isUser && (
          <Box className="assistant-avatar">
            <Sparkles size={16} />
          </Box>
        )}
        <Box className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          {renderMarkdown(text)}
        </Box>
        {showActions && <ChatMessageActions text={text} onRegenerate={onRegenerate} />}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function ChatInterface({ setView }) {
  const { session, token, getAuthToken } = useAuth();
  const userStorageScope = session?.user?.id || session?.user?.email || 'anonymous';
  const sessionStorageKey = `${SESSION_STORAGE_KEY}:${userStorageScope}`;
  const sessionIndexStorageKey = `${SESSION_INDEX_STORAGE_KEY}:${userStorageScope}`;
  const showPlotFixtureGallery = new URLSearchParams(window.location.search).has('plotTest');
  const [sessionId, setSessionId] = useState(() => {
    const existing = window.localStorage.getItem(sessionStorageKey);
    if (existing) {
      rememberSessionId(existing, sessionIndexStorageKey);
      return existing;
    }

    const cachedSessionId = readCachedSessions(userStorageScope)[0]?.session_id;
    const nextSessionId = cachedSessionId || createSessionId();
    window.localStorage.setItem(sessionStorageKey, nextSessionId);
    rememberSessionId(nextSessionId, sessionIndexStorageKey);
    return nextSessionId;
  });

  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [activeConversationId, setActiveConversationId] = useState(sessionId);
  const [messages, setMessages] = useState(() => {
    const cachedMessages = readCachedMessages(userStorageScope, sessionId).map((item) => toChatMessage(item, sessionId));
    return cachedMessages.length ? cachedMessages : [makeWelcomeMessage(sessionId)];
  });
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [chatSessions, setChatSessions] = useState(() => readCachedSessions(userStorageScope));
  const [composerText, setComposerText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const [sidebarState, setSidebarState] = useState('full'); // 'full' | 'mini' | 'closed'
  const selectedModelRef = useRef('');
  const messagesEndRef = useRef(null);
  const activeStreamControllerRef = useRef(null);
  const autoOpenedLatestSessionRef = useRef(false);
  const messagesRef = useRef(messages);

  const getAuthorizationHeaders = useCallback(async (headers = {}, options = {}) => {
    const authToken = token || await getAuthToken();
    if (!authToken) {
      if (options.required) {
        throw new Error('Your login session is still loading. Please wait a moment and try again.');
      }
      return headers;
    }
    return {
      ...headers,
      Authorization: `Bearer ${authToken}`,
    };
  }, [getAuthToken, token]);

  const getRequiredAuthorizationHeaders = useCallback((headers = {}) => (
    getAuthorizationHeaders(headers, { required: true })
  ), [getAuthorizationHeaders]);

  const loadSessionMessages = useCallback(async (targetSessionId) => {
    const params = new URLSearchParams({ limit: '200' });

    const headers = await getAuthorizationHeaders();
    if (!headers.Authorization) {
      const cachedMessages = readCachedMessages(userStorageScope, targetSessionId).map((item) => toChatMessage(item, targetSessionId));
      return cachedMessages.length ? cachedMessages : [makeWelcomeMessage(targetSessionId)];
    }

    return fetch(`${BACKEND_BASE}/chat/${encodeURIComponent(targetSessionId)}/messages?${params.toString()}`, {
      headers,
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch conversation history');
        return res.json();
      })
      .then((data) => {
        const rows = data?.messages || [];
        writeCachedMessages(userStorageScope, targetSessionId, rows);
        const loadedMessages = rows.map((item) => toChatMessage(item, targetSessionId));
        return loadedMessages.length ? loadedMessages : [makeWelcomeMessage(targetSessionId)];
      });
  }, [getAuthorizationHeaders, userStorageScope]);

  // Use a ref to track loadSessionMessages to avoid unnecessary re-runs
  const loadSessionMessagesRef = useRef(loadSessionMessages);
  loadSessionMessagesRef.current = loadSessionMessages;

  const refreshSessions = useCallback(async () => {
    const params = new URLSearchParams({ limit: '50' });

    const headers = await getAuthorizationHeaders();
    if (!headers.Authorization) {
      setChatSessions(readCachedSessions(userStorageScope));
      return undefined;
    }

    return fetch(`${BACKEND_BASE}/chat/sessions?${params.toString()}`, {
      headers,
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch chat sessions');
        return res.json();
      })
      .then((data) => {
        const sessions = data?.sessions || [];
        rememberSessionIds([sessionId, ...sessions.map((item) => item.session_id)], sessionIndexStorageKey);
        writeCachedSessions(userStorageScope, sessions);
        setChatSessions(sessions);

        const latestSessionId = sessions[0]?.session_id;
        const currentSessionExists = sessions.some((item) => item.session_id === sessionId);
        const currentHasConversation = messagesRef.current.some((message) => (
          message.id !== WELCOME_MESSAGE_ID && messageText(message).trim()
        ));
        const shouldOpenLatestSession = latestSessionId && (
          !currentSessionExists
          || (!currentHasConversation && latestSessionId !== sessionId)
        );
        if (!autoOpenedLatestSessionRef.current && shouldOpenLatestSession) {
          autoOpenedLatestSessionRef.current = true;
          window.localStorage.setItem(sessionStorageKey, latestSessionId);
          rememberSessionId(latestSessionId, sessionIndexStorageKey);
          setSessionId(latestSessionId);
          setActiveConversationId(latestSessionId);
        }
      })
      .catch((err) => {
        console.error('Failed to load chat sessions:', err);
        setChatSessions(readCachedSessions(userStorageScope));
      });
  }, [getAuthorizationHeaders, sessionId, sessionIndexStorageKey, sessionStorageKey, userStorageScope]);

  useEffect(() => {
    const cachedSessions = readCachedSessions(userStorageScope);
    if (cachedSessions.length) {
      setChatSessions(cachedSessions);
    }

    const cachedSessionId = window.localStorage.getItem(sessionStorageKey) || cachedSessions[0]?.session_id;
    if (cachedSessionId && cachedSessionId !== sessionId) {
      setSessionId(cachedSessionId);
      setActiveConversationId(cachedSessionId);
      return;
    }

    const targetSessionId = cachedSessionId || sessionId;
    const cachedMessages = readCachedMessages(userStorageScope, targetSessionId).map((item) => toChatMessage(item, targetSessionId));
    if (cachedMessages.length) {
      setMessages(cachedMessages);
      setHistoryLoaded(true);
    }
  }, [sessionId, sessionStorageKey, userStorageScope]);

  useEffect(() => {
    const cacheableMessages = messages
      .filter((message) => message.id !== WELCOME_MESSAGE_ID)
      .map((message) => ({
        id: message.id,
        role: message.role || (message.senderId === 'assistant' ? 'assistant' : 'user'),
        content: messageText(message),
        metadata: {},
        created_at:
          typeof message.createdAt === 'string'
            ? message.createdAt
            : (message.createdAt ? new Date(message.createdAt).toISOString() : new Date().toISOString()),
      }))
      .filter((message) => message.content.trim());

    if (cacheableMessages.length) {
      writeCachedMessages(userStorageScope, sessionId, cacheableMessages);
    }
  }, [messages, sessionId, userStorageScope]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    selectedModelRef.current = selectedModel;
  }, [selectedModel]);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    if (!session?.user?.id && !session?.user?.email) return;

    let cancelled = false;
    getAuthToken()
      .then(() => {
        if (!cancelled) refreshSessions();
      })
      .catch((err) => {
        console.error('Failed to refresh chat sessions after login:', err);
      });

    return () => {
      cancelled = true;
    };
  }, [getAuthToken, refreshSessions, session?.user?.email, session?.user?.id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

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

  useEffect(() => {
    let active = true;
    setHistoryLoaded(false);
    setActiveConversationId(sessionId);

    loadSessionMessagesRef.current(sessionId)
      .then((nextMessages) => {
        if (!active) return;
        setMessages(nextMessages);
        setHistoryLoaded(true);
      })
      .catch((err) => {
        console.error('Failed to load conversation history:', err);
        if (active) {
          const fallbackMessages = [makeWelcomeMessage(sessionId)];
          setMessages(fallbackMessages);
          setHistoryLoaded(true);
        }
      });

    return () => {
      active = false;
    };
  }, [sessionId, token, userStorageScope]);

  const activeSession = chatSessions.find((item) => item.session_id === sessionId);
  const activeTitle = activeSession?.title || 'Portfolio Assistant';

  const conversations = useMemo(() => {
    const sessionConversations = chatSessions.map((summary) => (
      makeConversation(summary.session_id, summary.session_id === sessionId ? messages : [], summary)
    ));
    return upsertConversation(sessionConversations, makeConversation(sessionId, messages, activeSession));
  }, [activeSession, chatSessions, messages, sessionId]);

  const adapter = useMemo(() => ({
    async sendMessage({ message, signal }) {
      const textPart = message.parts?.find(p => p.type === 'text');
      const userText = textPart ? textPart.text : (typeof message === 'string' ? message : '');
      const headers = await getRequiredAuthorizationHeaders({
        'Content-Type': 'application/json',
      });

      const response = await fetch(`${BACKEND_BASE}/chat/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ 
          session_id: sessionId, 
          user_message: userText,
          model: selectedModelRef.current || null
        }),
        signal,
      });

      if (!response.ok) {
        const detail = await readApiError(response, 'Network response was not ok');
        throw new Error(`Backend returned ${response.status}: ${detail}`);
      }

      const ndjsonStream = await parseNDJSONStream(response, signal);

      const transformStream = new TransformStream({
        transform(chunk, controller) {
          if (chunk.type === 'start') {
            controller.enqueue({ ...chunk, author: botUser });
          } else if (chunk.type === 'text-start') {
            controller.enqueue(chunk);
          } else if (chunk.type === 'text-end') {
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

      const refreshAfterStream = new TransformStream({
        flush() {
          refreshSessions();
        },
      });

      return ndjsonStream.pipeThrough(transformStream).pipeThrough(refreshAfterStream);
    },
  }), [getRequiredAuthorizationHeaders, refreshSessions, sessionId]);

  const handleMessagesChange = (nextMessages) => {
    const alignedMessages = nextMessages.map((message) => {
      const role = message.role || (message.senderId === 'assistant' ? 'assistant' : 'user');
      return {
        ...message,
        conversationId: message.conversationId || activeConversationId,
        role,
        author: message.author || (role === 'assistant' ? botUser : youUser),
        createdAt:
          typeof message.createdAt === 'string'
            ? message.createdAt
            : (message.createdAt ? new Date(message.createdAt).toISOString() : new Date().toISOString()),
      };
    });

    setMessages(alignedMessages);
  };

  const handleNewChat = useCallback(() => {
    autoOpenedLatestSessionRef.current = true;
    const nextSessionId = createSessionId();
    const nextMessages = [makeWelcomeMessage(nextSessionId)];

    window.localStorage.setItem(sessionStorageKey, nextSessionId);
    rememberSessionId(nextSessionId, sessionIndexStorageKey);

    setSessionId(nextSessionId);
    setActiveConversationId(nextSessionId);
    setMessages(nextMessages);
    setHistoryLoaded(true);
  }, [sessionIndexStorageKey, sessionStorageKey]);

  const handleActiveConversationChange = useCallback((nextId) => {
    console.log('handleActiveConversationChange called with:', nextId, 'current active:', activeConversationId);
    if (!nextId || nextId === activeConversationId) return;

    window.localStorage.setItem(sessionStorageKey, nextId);
    rememberSessionId(nextId, sessionIndexStorageKey);
    setSessionId(nextId);
    setActiveConversationId(nextId);
  }, [activeConversationId, sessionIndexStorageKey, sessionStorageKey]);

  const handleDeleteConversation = useCallback(async (event, targetSessionId) => {
    event.stopPropagation();
    if (!targetSessionId) return;

    try {
      const headers = await getRequiredAuthorizationHeaders();
      const response = await fetch(`${BACKEND_BASE}/chat/${encodeURIComponent(targetSessionId)}`, {
        method: 'DELETE',
        headers,
      });
      if (!response.ok) throw new Error('Failed to delete chat');

      forgetSessionId(targetSessionId, sessionIndexStorageKey);
      setChatSessions((current) => current.filter((item) => item.session_id !== targetSessionId));

      if (targetSessionId === sessionId) {
        handleNewChat();
      } else {
        refreshSessions();
      }
    } catch (error) {
      console.error('Failed to delete chat session:', error);
    }
  }, [getRequiredAuthorizationHeaders, handleNewChat, refreshSessions, sessionId, sessionIndexStorageKey]);

  const stopStreaming = useCallback(() => {
    activeStreamControllerRef.current?.abort();
    activeStreamControllerRef.current = null;
    setIsSubmitting(false);
    setStreamStatus('');
  }, []);

  const sendPrompt = useCallback(async (rawText) => {
    const userText = String(rawText || '').trim();
    if (!userText || isSubmitting) return;

    const now = new Date().toISOString();
    const userMessage = {
      id: `user-${Date.now()}`,
      conversationId: activeConversationId,
      role: 'user',
      author: youUser,
      createdAt: now,
      parts: [{ type: 'text', text: userText }],
    };
    const assistantId = `assistant-${Date.now() + 1}`;
    const assistantMessage = {
      id: assistantId,
      conversationId: activeConversationId,
      role: 'assistant',
      author: botUser,
      createdAt: now,
      status: 'streaming',
      parts: [{ type: 'text', text: '' }],
    };

    setComposerText('');
    setIsSubmitting(true);
    setStreamStatus('Starting stream');
    setMessages((current) => [...current, userMessage, assistantMessage]);

    const controller = new AbortController();
    activeStreamControllerRef.current = controller;

    try {
      const headers = await getRequiredAuthorizationHeaders({
        'Content-Type': 'application/json',
      });
      const response = await fetch(`${BACKEND_BASE}/chat/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          session_id: sessionId,
          user_message: userText,
          model: selectedModelRef.current || null,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await readApiError(response, 'Network response was not ok');
        throw new Error(`Backend returned ${response.status}: ${detail}`);
      }

      const reader = (await parseNDJSONStream(response, controller.signal)).getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (value.type === 'text-delta') {
          setStreamStatus('Streaming response');
          setMessages((current) => patchAssistantMessage(
            current,
            assistantId,
            (text) => text + (value.delta || ''),
          ));
        } else if (value.type === 'status') {
          setStreamStatus(value.label || value.stage || 'Processing');
        } else if (value.type === 'data-plot' && value.plotId) {
          setMessages((current) => patchAssistantMessage(
            current,
            assistantId,
            (text) => `${text}\n${PLOT_TOKEN}${value.plotId}`,
          ));
        } else if (value.type === 'finish') {
          setMessages((current) => current.map((message) => (
            message.id === assistantId
              ? {
                  ...message,
                  status: 'sent',
                  parts: message.parts.map((part) => (
                    part.type === 'text' ? { ...part, state: 'done' } : part
                  )),
                }
              : message
          )));
        }
      }

      refreshSessions();
    } catch (error) {
      if (controller.signal.aborted) {
        setMessages((current) => current.map((message) => (
          message.id === assistantId
            ? {
                ...message,
                status: 'stopped',
                parts: [{ type: 'text', text: `${messageText(message).trim()}\n\nResponse stopped.`.trim(), state: 'done' }],
              }
            : message
        )));
        return;
      }

      setMessages((current) => patchAssistantMessage(
        current,
        assistantId,
        () => `I could not reach the backend cleanly: ${error.message || 'request failed'}`,
      ));
    } finally {
      if (activeStreamControllerRef.current === controller) {
        activeStreamControllerRef.current = null;
      }
      setIsSubmitting(false);
      setStreamStatus('');
    }
  }, [activeConversationId, getRequiredAuthorizationHeaders, isSubmitting, refreshSessions, sessionId]);

  const handleComposerKeyDown = useCallback((event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendPrompt(composerText);
    }
  }, [composerText, sendPrompt]);

  const handleRegenerate = useCallback(() => {
    const lastUserMessage = [...messages].reverse().find((message) => message.role === 'user');
    const text = messageText(lastUserMessage);
    if (text) {
      sendPrompt(text);
    }
  }, [messages, sendPrompt]);

  if (showPlotFixtureGallery) {
    return <PlotFixtureGallery />;
  }

  return (
    <Box className={`chatgpt-shell sidebar-${sidebarState}`}>
      <Box className="chatgpt-sidebar">

        {/* ════════════════════════════════════════════════════════════════════
            FULL SIDEBAR CONTENT (260px)
            Fades out fast when collapsing, fades in with delay when expanding
            ════════════════════════════════════════════════════════════════════ */}
        <Box className="sidebar-full-content">
          <Box className="sidebar-header">
            <Typography className="sidebar-brand">
              Portfolio AI <span>Plus</span>
            </Typography>
            <Tooltip title="Collapse to icon rail">
              <IconButton
                className="sidebar-icon"
                aria-label="Collapse sidebar to icon rail"
                onClick={() => setSidebarState('mini')}
              >
                <PanelLeft size={18} />
              </IconButton>
            </Tooltip>
          </Box>

          <Button
            className="sidebar-new-chat"
            startIcon={<SquarePen size={16} />}
            onClick={handleNewChat}
            fullWidth
          >
            New chat
          </Button>

          <Box className="sidebar-section">
            <Typography className="sidebar-section-title">Tools</Typography>
            <button className="sidebar-tool" type="button" onClick={() => setView('analytics')}>
              <BarChart2 size={16} />
              Analytics Dashboard
            </button>
            <button className="sidebar-tool" type="button">
              <Search size={16} />
              Explore Sessions
            </button>
          </Box>

          <Box className="sidebar-section sidebar-recents">
            <Typography className="sidebar-section-title">Recents</Typography>
            {conversations.slice(0, 14).map((conversation) => (
              <div
                key={conversation.id}
                role="button"
                tabIndex={0}
                className={`recent-chat ${conversation.id === sessionId ? 'active' : ''}`}
                onClick={() => handleActiveConversationChange(conversation.id)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleActiveConversationChange(conversation.id);
                  }
                }}
              >
                <span>{conversation.title}</span>
                <Tooltip title="Delete chat">
                  <IconButton
                    className="recent-delete"
                    aria-label={`Delete ${conversation.title}`}
                    size="small"
                    onClick={(event) => handleDeleteConversation(event, conversation.id)}
                  >
                    <Trash2 size={14} />
                  </IconButton>
                </Tooltip>
              </div>
            ))}
          </Box>

          <Box className="sidebar-profile">
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  avatarBox: {
                    width: '36px',
                    height: '36px',
                  },
                },
              }}
            />
            <Box className="profile-copy">
              <Typography className="profile-name">
                {session?.user?.name || 'Portfolio Governance'}
              </Typography>
              <Typography className="profile-plan">
                {session?.user?.plan || 'Advisory workspace'}
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* ════════════════════════════════════════════════════════════════════
            MINI ICON-RAIL CONTENT (64px)
            Inspired by toolpad-master DashboardLayout mini drawer:
            - Icon-only with MUI Tooltip on hover (placement="right")
            - Session initials in avatar buttons
            - Active session highlighted with inner border glow
            ════════════════════════════════════════════════════════════════════ */}
        <Box className="sidebar-mini-content">
          {/* Toggle: expand back to full */}
          <Tooltip title="Expand sidebar" placement="right" arrow>
            <IconButton
              className="mini-icon-btn"
              aria-label="Expand sidebar"
              onClick={() => setSidebarState('full')}
            >
              <PanelLeft size={17} />
            </IconButton>
          </Tooltip>

          {/* New chat */}
          <Tooltip title="New chat" placement="right" arrow>
            <IconButton
              className="mini-icon-btn"
              aria-label="New chat"
              onClick={handleNewChat}
            >
              <SquarePen size={16} />
            </IconButton>
          </Tooltip>

          <Box className="mini-divider" />

          {/* Tools */}
          <Tooltip title="Analytics Dashboard" placement="right" arrow>
            <IconButton
              className="mini-icon-btn"
              aria-label="Analytics Dashboard"
              onClick={() => setView('analytics')}
            >
              <BarChart2 size={16} />
            </IconButton>
          </Tooltip>

          <Tooltip title="Explore Sessions" placement="right" arrow>
            <IconButton
              className="mini-icon-btn"
              aria-label="Explore Sessions"
            >
              <Search size={16} />
            </IconButton>
          </Tooltip>

          <Box className="mini-divider" />

          {/* Spacer — no chat history in mini mode */}
          <Box sx={{ flex: 1 }} />

          {/* Profile avatar */}
          <Tooltip
            title={session?.user?.name || 'Profile'}
            placement="right"
            arrow
          >
            <Box className="mini-profile-user-button">
              <UserButton
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    avatarBox: {
                      width: '38px',
                      height: '38px',
                    },
                  },
                }}
              />
            </Box>
          </Tooltip>
        </Box>
      </Box>

        <Box className="chatgpt-main">
        <Box className="chatgpt-topbar">
          <Box className="topbar-title-wrap">
            <Typography className="topbar-title">{activeTitle}</Typography>
            <Typography className="topbar-status">
              {historyLoaded ? 'Memory loaded' : 'Loading memory...'}
            </Typography>
          </Box>
          <Box className="topbar-actions">
            <Button
              className="topbar-dashboard"
              startIcon={<BarChart2 size={16} />}
              onClick={() => setView('analytics')}
            >
              Analytics
            </Button>
            <Button className="share-button" startIcon={<Share size={18} />}>
              Share
            </Button>
            <Tooltip title="More">
              <IconButton className="topbar-icon" aria-label="More options">
                <MoreHorizontal size={22} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <Box className="chat-scroll">
          <Box className="message-column">
            {messages.map((message) => (
              <ChatMessageRow
                key={message.id}
                message={message}
                onRegenerate={handleRegenerate}
              />
            ))}
            {isSubmitting && (
              <Box className="streaming-status">
                <Sparkles size={15} />
                {streamStatus || 'Thinking through the governance context'}
              </Box>
            )}
            <div ref={messagesEndRef} />
          </Box>
        </Box>

        <Box className="composer-shell">
          <Box className="composer-card">
            <CustomAttachButtonWithModelSelector
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              availableModels={availableModels}
              loadingModels={loadingModels}
            />
            <textarea
              className="composer-input"
              value={composerText}
              onChange={(event) => setComposerText(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Ask anything"
              rows={1}
            />
            <Box className="composer-side-actions">
              <Typography className="thinking-label">Thinking</Typography>
              <ChevronDown size={16} />
              <Tooltip title="Voice input">
                <IconButton className="composer-icon-button" aria-label="Voice input">
                  <Mic size={21} />
                </IconButton>
              </Tooltip>
              <Tooltip title={isSubmitting ? 'Stop response' : 'Send'}>
                <span>
                  <IconButton
                    className={`composer-send-button ${isSubmitting ? 'composer-stop-button' : ''}`}
                    aria-label={isSubmitting ? 'Stop response' : 'Send message'}
                    disabled={!composerText.trim() && !isSubmitting}
                    onClick={() => (isSubmitting ? stopStreaming() : sendPrompt(composerText))}
                  >
                    {isSubmitting ? <Square size={18} fill="currentColor" /> : <AudioLines size={22} />}
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
          </Box>
          <Typography className="composer-disclaimer">
            Portfolio AI can make mistakes. Check important governance outputs.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
