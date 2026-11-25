import React from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { HeroGreeting } from './components/HeroGreeting';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { SettingsModal } from './components/SettingsModal';
import type { ChatSession, Message, ModelMode, AppSettings } from './types';
import {
  loadSessions,
  saveSessions,
  loadActiveSessionId,
  saveActiveSessionId,
  loadSettings,
  saveSettings,
} from './services/storage';
import { streamChatCompletion, checkBackendHealth } from './services/api';

export function App() {
  const [sessions, setSessions] = React.useState<ChatSession[]>(() => loadSessions());
  const [activeSessionId, setActiveSessionId] = React.useState<string | null>(() => loadActiveSessionId());
  const [selectedMode, setSelectedMode] = React.useState<ModelMode>('tutor');
  const [input, setInput] = React.useState('');
  const [isStreaming, setIsStreaming] = React.useState(false);
  const [sidebarOpen, setSidebarOpen] = React.useState(true);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [settings, setSettings] = React.useState<AppSettings>(() => loadSettings());
  const [backendOnline, setBackendOnline] = React.useState(false);

  const abortControllerRef = React.useRef<AbortController | null>(null);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Sync sessions to localStorage
  React.useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  // Sync active session
  React.useEffect(() => {
    if (activeSessionId) {
      saveActiveSessionId(activeSessionId);
    }
  }, [activeSessionId]);

  // Check backend health on mount and every 30s
  React.useEffect(() => {
    let mounted = true;
    const verifyHealth = async () => {
      const res = await checkBackendHealth(settings.apiUrl);
      if (mounted) setBackendOnline(res.ok);
    };
    verifyHealth();
    const interval = setInterval(verifyHealth, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [settings.apiUrl]);

  // Scroll to bottom when messages change or streaming
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const messages = React.useMemo(
    () => (activeSession ? activeSession.messages : []),
    [activeSession]
  );

  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  const handleNewChat = () => {
    setActiveSessionId(null);
    setInput('');
  };

  const handleSelectSession = (id: string) => {
    setActiveSessionId(id);
    const session = sessions.find((s) => s.id === id);
    if (session) {
      setSelectedMode(session.mode || 'tutor');
    }
  };

  const handleRenameSession = (id: string, newTitle: string) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, title: newTitle, updatedAt: Date.now() } : s))
    );
  };

  const handleDeleteSession = (id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      setActiveSessionId(null);
    }
  };

  const handleClearAllSessions = () => {
    setSessions([]);
    setActiveSessionId(null);
  };

  const handleSaveSettings = (newSettings: AppSettings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  const handleSend = async (customPrompt?: string, customMode?: ModelMode) => {
    const textToSend = customPrompt || input;
    if (!textToSend.trim() || isStreaming) return;

    const modeToUse = customMode || selectedMode;
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now() + 1}`;

    const userMessage: Message = {
      id: userMsgId,
      role: 'user',
      content: textToSend.trim(),
      timestamp: Date.now(),
      mode: modeToUse,
    };

    const initialAssistantMessage: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      mode: modeToUse,
      status: 'streaming',
    };

    let targetSessionId = activeSessionId;

    if (!targetSessionId) {
      // Create new session
      const title = textToSend.trim().slice(0, 36) + (textToSend.trim().length > 36 ? '...' : '');
      const newSession: ChatSession = {
        id: `session-${Date.now()}`,
        title,
        messages: [userMessage, initialAssistantMessage],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        mode: modeToUse,
      };
      targetSessionId = newSession.id;
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
    } else {
      // Append to active session
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === targetSessionId) {
            return {
              ...s,
              messages: [...s.messages, userMessage, initialAssistantMessage],
              updatedAt: Date.now(),
            };
          }
          return s;
        })
      );
    }

    setInput('');
    setIsStreaming(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const sessionMessages = activeSession
      ? [...activeSession.messages, userMessage]
      : [userMessage];

    await streamChatCompletion(
      sessionMessages,
      modeToUse,
      settings,
      {
        onChunk: (chunk) => {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === targetSessionId) {
                const updatedMsgs = s.messages.map((m) => {
                  if (m.id === assistantMsgId) {
                    return { ...m, content: m.content + chunk };
                  }
                  return m;
                });
                return { ...s, messages: updatedMsgs };
              }
              return s;
            })
          );
        },
        onDone: (full) => {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === targetSessionId) {
                const updatedMsgs = s.messages.map((m) => {
                  if (m.id === assistantMsgId) {
                    return { ...m, content: full || m.content, status: 'complete' as const };
                  }
                  return m;
                });
                return { ...s, messages: updatedMsgs };
              }
              return s;
            })
          );
          setIsStreaming(false);
          abortControllerRef.current = null;
        },
        onError: (err) => {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === targetSessionId) {
                const updatedMsgs = s.messages.map((m) => {
                  if (m.id === assistantMsgId) {
                    return {
                      ...m,
                      content:
                        m.content +
                        `\n\n> ⚠️ **Error:** ${err.message || 'Unable to connect to inference server.'}\n> Ensure Modal inference server is running.`,
                      status: 'error' as const,
                    };
                  }
                  return m;
                });
                return { ...s, messages: updatedMsgs };
              }
              return s;
            })
          );
          setIsStreaming(false);
          abortControllerRef.current = null;
        },
      },
      abortController.signal
    );
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#131314] text-[#e3e3e3]">
      {/* Gemini Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onCloseMobile={() => setSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
      />

      {/* Main Conversation Canvas */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        <Header
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          selectedMode={selectedMode}
          onSelectMode={setSelectedMode}
          onOpenSettings={() => setSettingsOpen(true)}
          backendOnline={backendOnline}
        />

        {/* Scrollable Conversation Stream */}
        <main className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <HeroGreeting
              onSelectPrompt={(prompt, mode) => {
                setSelectedMode(mode);
                handleSend(prompt, mode);
              }}
            />
          ) : (
            <div className="max-w-3xl mx-auto divide-y divide-[#282a2c]/40 pb-6">
              {messages.map((msg, idx) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  isStreaming={isStreaming && idx === messages.length - 1 && msg.role === 'assistant'}
                />
              ))}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          )}
        </main>

        {/* Floating Input Area */}
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={() => handleSend()}
          onStop={handleStop}
          isStreaming={isStreaming}
          selectedMode={selectedMode}
          onSelectMode={setSelectedMode}
        />
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onSaveSettings={handleSaveSettings}
        onClearAllSessions={handleClearAllSessions}
      />
    </div>
  );
}

export default App;
