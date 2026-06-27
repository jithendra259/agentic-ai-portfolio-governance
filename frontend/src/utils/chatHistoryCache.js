const CACHE_PREFIX = 'portfolio-ai-chat-cache';

function getStorage(storage) {
  if (storage) return storage;
  if (typeof window !== 'undefined' && window.localStorage) return window.localStorage;
  return null;
}

function safeUserScope(userId) {
  return encodeURIComponent(String(userId || 'anonymous'));
}

function safeSessionScope(sessionId) {
  return encodeURIComponent(String(sessionId || ''));
}

function readArray(key, storage) {
  const targetStorage = getStorage(storage);
  if (!targetStorage || !key) return [];

  try {
    const parsed = JSON.parse(targetStorage.getItem(key) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function writeArray(key, rows, storage) {
  const targetStorage = getStorage(storage);
  if (!targetStorage || !key) return;
  targetStorage.setItem(key, JSON.stringify(Array.isArray(rows) ? rows : []));
}

export function chatHistoryCacheKeys(userId) {
  const userScope = safeUserScope(userId);
  return {
    sessions: `${CACHE_PREFIX}:sessions:${userScope}`,
    messagesPrefix: `${CACHE_PREFIX}:messages:${userScope}:`,
  };
}

export function readCachedSessions(userId, storage) {
  return readArray(chatHistoryCacheKeys(userId).sessions, storage);
}

export function writeCachedSessions(userId, sessions, storage) {
  writeArray(chatHistoryCacheKeys(userId).sessions, sessions, storage);
}

export function readCachedMessages(userId, sessionId, storage) {
  const { messagesPrefix } = chatHistoryCacheKeys(userId);
  return readArray(`${messagesPrefix}${safeSessionScope(sessionId)}`, storage);
}

export function writeCachedMessages(userId, sessionId, messages, storage) {
  const { messagesPrefix } = chatHistoryCacheKeys(userId);
  writeArray(`${messagesPrefix}${safeSessionScope(sessionId)}`, messages, storage);
}
