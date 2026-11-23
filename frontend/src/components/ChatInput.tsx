import React from 'react';
import { ArrowUp, Square, Sparkles, GraduationCap, HelpCircle, SearchCheck, HeartHandshake } from 'lucide-react';
import { type ModelMode, MODES } from '../types';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  selectedMode: ModelMode;
  onSelectMode: (mode: ModelMode) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  onSend,
  onStop,
  isStreaming,
  selectedMode,
  onSelectMode,
}) => {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  React.useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming && input.trim()) {
        onSend();
      }
    }
  };

  const modePills = [
    { mode: 'general' as ModelMode, label: 'General', icon: Sparkles },
    { mode: 'tutor' as ModelMode, label: 'Tutor', icon: GraduationCap },
    { mode: 'quiz' as ModelMode, label: 'Quiz', icon: HelpCircle },
    { mode: 'diagnostic' as ModelMode, label: 'Diagnostic', icon: SearchCheck },
    { mode: 'companion' as ModelMode, label: 'Companion', icon: HeartHandshake },
  ];

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4">
      {/* Floating Gemini Input Container */}
      <div className="rounded-3xl bg-[#1e1f20] border border-[#333538] focus-within:border-[#4d5156] transition-all shadow-xl p-2.5">
        {/* Mode Quick Toggle Pills */}
        <div className="flex items-center gap-1.5 px-2 pt-1 pb-2 overflow-x-auto no-scrollbar">
          {modePills.map((p) => {
            const Icon = p.icon;
            const isSelected = selectedMode === p.mode;
            return (
              <button
                key={p.mode}
                type="button"
                onClick={() => onSelectMode(p.mode)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                  isSelected
                    ? 'bg-[#333538] text-white shadow-sm border border-indigo-500/40'
                    : 'text-[#8e918f] hover:text-[#c4c7c5] hover:bg-[#282a2c]'
                }`}
              >
                <Icon className={`w-3 h-3 ${isSelected ? 'text-indigo-400' : 'text-[#8e918f]'}`} />
                <span>{p.label}</span>
              </button>
            );
          })}
        </div>

        {/* Textarea & Actions Row */}
        <div className="flex items-end gap-2 px-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask StudyVerse (${MODES[selectedMode].name})...`}
            rows={1}
            className="flex-1 bg-transparent text-sm text-[#e3e3e3] placeholder-[#8e918f] resize-none outline-none max-h-44 py-1 leading-relaxed"
          />

          {/* Action Button: Send or Stop */}
          {isStreaming ? (
            <button
              onClick={onStop}
              className="p-2 rounded-full bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 transition-all shrink-0 shadow-sm"
              title="Stop response"
            >
              <Square className="w-4 h-4 fill-current" />
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={!input.trim()}
              className={`p-2 rounded-full transition-all shrink-0 ${
                input.trim()
                  ? 'bg-gradient-to-tr from-blue-500 to-indigo-500 text-white shadow-md shadow-indigo-500/30 hover:opacity-90 cursor-pointer'
                  : 'bg-[#282a2c] text-[#555] cursor-not-allowed'
              }`}
              title="Send message"
            >
              <ArrowUp className="w-4 h-4 stroke-[2.5]" />
            </button>
          )}
        </div>
      </div>

      {/* Footer Disclaimer */}
      <p className="text-[11px] text-[#8e918f] text-center mt-2.5 select-none">
        StudyVerse Qwen 2.5 3B · Fine-tuned on CISCE curriculum · Serverless inference on Modal GPU
      </p>
    </div>
  );
};
