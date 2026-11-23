import type { ChatSession, AppSettings } from '../types';

const STORAGE_KEYS = {
  CHATS: 'studyverse_gemini_chats',
  ACTIVE_CHAT: 'studyverse_gemini_active_chat',
  SETTINGS: 'studyverse_gemini_settings',
};

export const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: 'https://rockefashy--studyverse-qwen-serve-modelserver-serve.us-east.modal.run/v1',
  model: 'studyverse-qwen-2.5-3b',
  temperature: 0.2,
  maxTokens: 512,
};

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.SETTINGS);
    if (raw) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
    }
  } catch (err) {
    console.error('Failed to load settings:', err);
  }
  return DEFAULT_SETTINGS;
}

export function saveSettings(settings: AppSettings): void {
  try {
    localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
  } catch (err) {
    console.error('Failed to save settings:', err);
  }
}

export function loadSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.CHATS);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch (err) {
    console.error('Failed to load sessions:', err);
  }
  return [];
}

export function saveSessions(sessions: ChatSession[]): void {
  try {
    localStorage.setItem(STORAGE_KEYS.CHATS, JSON.stringify(sessions));
  } catch (err) {
    console.error('Failed to save sessions:', err);
  }
}

export function loadActiveSessionId(): string | null {
  return localStorage.getItem(STORAGE_KEYS.ACTIVE_CHAT);
}

export function saveActiveSessionId(id: string): void {
  localStorage.setItem(STORAGE_KEYS.ACTIVE_CHAT, id);
}
