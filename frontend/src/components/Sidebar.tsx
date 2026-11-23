import React from 'react';
import { Plus, MessageSquare, Trash2, Edit2, Check, X, Sparkles, Cpu } from 'lucide-react';
import type { ChatSession } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onCloseMobile: () => void;
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onCloseMobile,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onRenameSession,
}) => {
  const [editingId, setEditingId] = React.useState<string | null>(null);
  const [editTitle, setEditTitle] = React.useState('');

  const handleStartRename = (e: React.MouseEvent, s: ChatSession) => {
    e.stopPropagation();
    setEditingId(s.id);
    setEditTitle(s.title);
  };

  const handleSaveRename = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (editTitle.trim()) {
      onRenameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 md:hidden"
          onClick={onCloseMobile}
        />
      )}

      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 bg-[#1e1f20] border-r border-[#282a2c] flex flex-col transition-all duration-300 ease-in-out ${
          isOpen ? 'w-72 translate-x-0' : '-translate-x-full md:translate-x-0 md:w-0 md:border-r-0 md:overflow-hidden'
        }`}
      >
        <div className="p-3">
          {/* New Chat Button */}
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-full bg-[#131314] hover:bg-[#282a2c] border border-[#333538] text-sm font-medium text-[#e3e3e3] hover:text-white transition-all shadow-sm group"
          >
            <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-blue-500 to-rose-400 p-[1px] flex items-center justify-center">
              <div className="w-full h-full bg-[#131314] rounded-full flex items-center justify-center group-hover:bg-[#282a2c]">
                <Plus className="w-3.5 h-3.5 text-blue-300" />
              </div>
            </div>
            <span>New chat</span>
          </button>
        </div>

        {/* Recent Chats Section */}
        <div className="flex-1 overflow-y-auto px-2 py-1 space-y-1">
          <div className="px-3 py-2 text-[11px] font-semibold text-[#8e918f] uppercase tracking-wider">
            Recent Chats
          </div>

          {sessions.length === 0 ? (
            <div className="px-3 py-6 text-center text-xs text-[#8e918f]">
              <Sparkles className="w-6 h-6 mx-auto mb-2 opacity-30 text-indigo-400" />
              No conversations yet.<br />Start by saying hello!
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              const isEditing = editingId === session.id;

              return (
                <div
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={`group relative flex items-center gap-2.5 px-3 py-2.5 rounded-full text-xs cursor-pointer transition-colors ${
                    isActive
                      ? 'bg-[#282a2c] text-white font-medium shadow-inner'
                      : 'text-[#c4c7c5] hover:bg-[#282a2c]/60 hover:text-white'
                  }`}
                >
                  <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-indigo-400' : 'text-[#8e918f]'}`} />

                  {isEditing ? (
                    <div className="flex-1 flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        className="flex-1 bg-[#131314] text-xs px-2 py-0.5 rounded border border-indigo-500 text-white outline-none"
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveRename(e as unknown as React.MouseEvent, session.id);
                          if (e.key === 'Escape') handleCancelRename(e as unknown as React.MouseEvent);
                        }}
                      />
                      <button
                        onClick={(e) => handleSaveRename(e, session.id)}
                        className="p-1 hover:text-emerald-400"
                        title="Save"
                      >
                        <Check className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={handleCancelRename}
                        className="p-1 hover:text-rose-400"
                        title="Cancel"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <span className="flex-1 truncate">{session.title}</span>

                      {/* Actions on hover */}
                      <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 text-[#8e918f] transition-opacity">
                        <button
                          onClick={(e) => handleStartRename(e, session)}
                          className="p-1 hover:text-white transition-colors"
                          title="Rename"
                        >
                          <Edit2 className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                          }}
                          className="p-1 hover:text-rose-400 transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer info pill */}
        <div className="p-3 border-t border-[#282a2c] bg-[#1a1b1c]/80 text-[11px] text-[#8e918f] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Tesla T4 · Modal</span>
          </div>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
            $0 Idle
          </span>
        </div>
      </aside>
    </>
  );
};
