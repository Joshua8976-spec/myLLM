import React from 'react';
import { X, Check, Activity, Trash2, RotateCcw } from 'lucide-react';
import type { AppSettings } from '../types';
import { DEFAULT_SETTINGS } from '../services/storage';
import { checkBackendHealth } from '../services/api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSaveSettings: (settings: AppSettings) => void;
  onClearAllSessions: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSaveSettings,
  onClearAllSessions,
}) => {
  const [form, setForm] = React.useState<AppSettings>(settings);
  const [testing, setTesting] = React.useState(false);
  const [testResult, setTestResult] = React.useState<{ ok: boolean; message: string } | null>(null);

  React.useEffect(() => {
    setForm(settings);
    setTestResult(null);
  }, [settings, isOpen]);

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    const res = await checkBackendHealth(form.apiUrl);
    setTesting(false);
    if (res.ok) {
      setTestResult({
        ok: true,
        message: `Connected successfully! Model: ${res.model} (${res.latencyMs}ms)`,
      });
    } else {
      setTestResult({
        ok: false,
        message: `Failed to connect: ${res.error || 'Server unreachable'}`,
      });
    }
  };

  const handleSave = () => {
    onSaveSettings(form);
    onClose();
  };

  const handleReset = () => {
    setForm(DEFAULT_SETTINGS);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-lg bg-[#1e1f20] border border-[#333538] rounded-3xl p-6 shadow-2xl relative text-[#e3e3e3]">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-[#282a2c] mb-5">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">Settings</h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
              Inference Config
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full hover:bg-[#282a2c] text-[#8e918f] hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4 text-sm">
          {/* API URL */}
          <div>
            <label className="block text-xs font-medium text-[#c4c7c5] mb-1.5">
              Backend Inference URL (OpenAI Compatible)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={form.apiUrl}
                onChange={(e) => setForm({ ...form, apiUrl: e.target.value })}
                className="flex-1 bg-[#131314] border border-[#333538] focus:border-indigo-500 rounded-xl px-3.5 py-2 text-xs font-mono text-[#e3e3e3] outline-none"
                placeholder="https://...modal.run/v1"
              />
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={testing}
                className="px-3 py-2 rounded-xl bg-[#282a2c] hover:bg-[#333538] text-xs font-medium text-[#e3e3e3] border border-[#333538] transition-colors flex items-center gap-1.5 shrink-0"
              >
                <Activity className={`w-3.5 h-3.5 ${testing ? 'animate-spin text-indigo-400' : 'text-emerald-400'}`} />
                <span>{testing ? 'Testing...' : 'Test'}</span>
              </button>
            </div>
            {testResult && (
              <p className={`text-xs mt-1.5 ${testResult.ok ? 'text-emerald-400' : 'text-rose-400'}`}>
                {testResult.message}
              </p>
            )}
          </div>

          {/* Model Name */}
          <div>
            <label className="block text-xs font-medium text-[#c4c7c5] mb-1.5">
              Model Identifier
            </label>
            <input
              type="text"
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
              className="w-full bg-[#131314] border border-[#333538] focus:border-indigo-500 rounded-xl px-3.5 py-2 text-xs font-mono text-[#e3e3e3] outline-none"
            />
          </div>

          {/* Temperature Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-[#c4c7c5]">
                Temperature ({form.temperature})
              </label>
              <span className="text-[11px] text-[#8e918f]">Lower = deterministic, Higher = creative</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={form.temperature}
              onChange={(e) => setForm({ ...form, temperature: parseFloat(e.target.value) })}
              className="w-full accent-indigo-500 cursor-pointer"
            />
          </div>

          {/* Max Tokens Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-[#c4c7c5]">
                Max Tokens ({form.maxTokens})
              </label>
              <span className="text-[11px] text-[#8e918f]">Tokens generated per response</span>
            </div>
            <input
              type="range"
              min="128"
              max="1024"
              step="64"
              value={form.maxTokens}
              onChange={(e) => setForm({ ...form, maxTokens: parseInt(e.target.value, 10) })}
              className="w-full accent-indigo-500 cursor-pointer"
            />
          </div>

          {/* Danger Zone: Clear History */}
          <div className="pt-2 border-t border-[#282a2c] flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-[#c4c7c5]">Clear Conversation History</div>
              <div className="text-[11px] text-[#8e918f]">Deletes all stored chat sessions from your browser</div>
            </div>
            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to delete all chat history?')) {
                  onClearAllSessions();
                  onClose();
                }
              }}
              className="px-3 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 text-xs font-medium transition-colors flex items-center gap-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear History</span>
            </button>
          </div>
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-[#282a2c]">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs text-[#8e918f] hover:text-[#c4c7c5] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset to defaults</span>
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl hover:bg-[#282a2c] text-xs font-medium text-[#c4c7c5] transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-medium text-white transition-colors flex items-center gap-1.5 shadow-md shadow-indigo-600/20"
            >
              <Check className="w-3.5 h-3.5" />
              <span>Save Changes</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
