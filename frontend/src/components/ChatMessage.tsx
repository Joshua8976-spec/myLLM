import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Sparkles, Copy, Check } from 'lucide-react';
import { type Message, MODES } from '../types';

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, isStreaming }) => {
  const [copied, setCopied] = React.useState(false);
  const isUser = message.role === 'user';
  const modeConfig = message.mode ? MODES[message.mode] : null;

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="py-4 px-4 flex justify-end">
        <div className="max-w-2xl bg-[#282a2c] text-[#e3e3e3] rounded-3xl rounded-tr-sm px-5 py-3.5 shadow-sm border border-[#333538] text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="py-5 px-4 sm:px-6 hover:bg-[#1a1b1c]/40 transition-colors group">
      <div className="max-w-3xl mx-auto flex gap-4 items-start">
        {/* Gemini Avatar */}
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 via-indigo-500 to-rose-400 p-[1.5px] shrink-0 mt-0.5 shadow-md shadow-indigo-500/20">
          <div className="w-full h-full bg-[#131314] rounded-full flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-transparent bg-clip-text bg-gradient-to-tr from-blue-400 to-rose-300" />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs font-semibold text-[#e3e3e3]">StudyVerse AI</span>
            {modeConfig && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#1e1f20] border border-[#333538] text-[#8e918f] font-mono">
                {modeConfig.label}
              </span>
            )}
          </div>

          {/* Markdown & LaTeX renderer */}
          <div className="prose prose-invert prose-sm max-w-none text-[#e3e3e3] leading-relaxed break-words">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                code: ({ className, children, ...props }) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const codeString = String(children).replace(/\n$/, '');
                  const isInline = !match && !codeString.includes('\n');

                  if (isInline) {
                    return (
                      <code className="px-1.5 py-0.5 rounded bg-[#282a2c] text-indigo-300 font-mono text-[12px] border border-[#333538]" {...props}>
                        {children}
                      </code>
                    );
                  }

                  return (
                    <div className="relative my-3 rounded-xl overflow-hidden border border-[#333538] bg-[#1a1b1c]">
                      <div className="flex items-center justify-between px-3 py-1.5 bg-[#282a2c]/60 border-b border-[#333538] text-[11px] text-[#8e918f] font-mono">
                        <span>{match ? match[1] : 'code'}</span>
                        <button
                          onClick={() => navigator.clipboard.writeText(codeString)}
                          className="hover:text-white transition-colors"
                          title="Copy Code"
                        >
                          <Copy className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <pre className="p-3.5 overflow-x-auto text-xs text-[#c4c7c5] font-mono bg-transparent m-0">
                        <code>{codeString}</code>
                      </pre>
                    </div>
                  );
                },
                ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-indigo-500/60 pl-3.5 my-2 text-[#8e918f] italic">
                    {children}
                  </blockquote>
                ),
                table: ({ children }) => (
                  <div className="overflow-x-auto my-3 border border-[#333538] rounded-lg">
                    <table className="min-w-full divide-y divide-[#333538] text-xs">{children}</table>
                  </div>
                ),
                th: ({ children }) => <th className="px-3 py-2 text-left font-semibold bg-[#282a2c] text-[#e3e3e3]">{children}</th>,
                td: ({ children }) => <td className="px-3 py-2 border-t border-[#333538] text-[#c4c7c5]">{children}</td>,
              }}
            >
              {message.content}
            </ReactMarkdown>

            {/* Blinking cursor while streaming */}
            {isStreaming && <span className="typing-cursor" />}
          </div>

          {/* Message action bar */}
          {!isStreaming && message.content && (
            <div className="flex items-center gap-2 mt-3 pt-2 opacity-0 group-hover:opacity-100 transition-opacity text-[#8e918f]">
              <button
                onClick={handleCopy}
                className="p-1.5 rounded-lg hover:bg-[#282a2c] hover:text-[#e3e3e3] transition-colors flex items-center gap-1 text-xs"
                title="Copy response"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span className="text-[11px]">{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
