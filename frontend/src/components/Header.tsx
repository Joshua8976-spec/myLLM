import React from 'react';
import { Menu, Sparkles, Settings as SettingsIcon, ChevronDown, Check } from 'lucide-react';
import { type ModelMode, MODES } from '../types';

interface HeaderProps {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  selectedMode: ModelMode;
  onSelectMode: (mode: ModelMode) => void;
  onOpenSettings: () => void;
  backendOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  sidebarOpen,
  onToggleSidebar,
  selectedMode,
  onSelectMode,
  onOpenSettings,
  backendOnline,
}) => {
  const [dropdownOpen, setDropdownOpen] = React.useState(false);
  const currentMode = MODES[selectedMode];

  return (
    <header className="h-16 px-4 border-b border-[#282a2c] bg-[#131314]/90 backdrop-blur-md flex items-center justify-between sticky top-0 z-30 select-none">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-full hover:bg-[#282a2c] text-[#c4c7c5] hover:text-white transition-colors"
          title={sidebarOpen ? 'Collapse menu' : 'Expand menu'}
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Gemini Sparkle Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 via-indigo-500 to-rose-400 p-[1.5px] flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-[#131314] rounded-full flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-transparent bg-clip-text bg-gradient-to-tr from-blue-400 to-rose-300" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-base text-[#e3e3e3] tracking-tight">StudyVerse</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-mono">
                Qwen 2.5 3B
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2.5">
        {/* Mode Selector Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#1e1f20] hover:bg-[#282a2c] border border-[#333538] text-xs font-medium text-[#e3e3e3] transition-all"
          >
            <span className={`w-2 h-2 rounded-full bg-gradient-to-r ${currentMode.gradient}`} />
            <span>{currentMode.name}</span>
            <ChevronDown className="w-3.5 h-3.5 text-[#8e918f]" />
          </button>

          {dropdownOpen && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setDropdownOpen(false)} />
              <div className="absolute right-0 mt-2 w-72 rounded-2xl bg-[#1e1f20] border border-[#333538] shadow-2xl p-2 z-50 animate-in fade-in zoom-in-95 duration-100">
                <div className="px-3 py-2 text-[11px] font-semibold text-[#8e918f] uppercase tracking-wider">
                  Select Task Mode
                </div>
                {(Object.keys(MODES) as ModelMode[]).map((modeKey) => {
                  const m = MODES[modeKey];
                  const isSelected = selectedMode === modeKey;
                  return (
                    <button
                      key={modeKey}
                      onClick={() => {
                        onSelectMode(modeKey);
                        setDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2.5 rounded-xl flex items-start justify-between gap-2 transition-colors ${
                        isSelected ? 'bg-indigo-500/15 text-white' : 'hover:bg-[#282a2c] text-[#c4c7c5]'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full bg-gradient-to-r ${m.gradient}`} />
                          <span className="text-xs font-medium">{m.name}</span>
                        </div>
                        <p className="text-[11px] text-[#8e918f] mt-0.5 leading-snug">{m.description}</p>
                      </div>
                      {isSelected && <Check className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />}
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* Backend Status indicator */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#1e1f20] border border-[#333538] text-xs text-[#c4c7c5]"
          title={backendOnline ? 'Connected to Modal GPU inference' : 'Connecting to Modal backend...'}
        >
          <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
          <span className="text-[11px] text-[#8e918f] hidden sm:inline">{backendOnline ? 'Modal Live' : 'Connecting'}</span>
        </div>

        {/* Settings button */}
        <button
          onClick={onOpenSettings}
          className="p-2 rounded-full hover:bg-[#282a2c] text-[#c4c7c5] hover:text-white transition-colors"
          title="Endpoint Settings"
        >
          <SettingsIcon className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
};
