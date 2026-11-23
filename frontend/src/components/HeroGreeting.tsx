import React from 'react';
import { Sparkles, GraduationCap, HelpCircle, SearchCheck, HeartHandshake } from 'lucide-react';
import type { ModelMode } from '../types';

interface HeroGreetingProps {
  onSelectPrompt: (prompt: string, mode: ModelMode) => void;
}

export const HeroGreeting: React.FC<HeroGreetingProps> = ({ onSelectPrompt }) => {
  const suggestions = [
    {
      title: 'Explain Newton\'s Second Law',
      subtitle: 'Derives F = ma for constant mass with LaTeX math',
      mode: 'tutor' as ModelMode,
      icon: GraduationCap,
      prompt: 'Explain Newton\'s Second Law and derive F = ma for constant mass.',
      color: 'from-blue-500/20 to-indigo-500/10 hover:border-blue-500/40',
      iconColor: 'text-blue-400',
    },
    {
      title: 'Coulomb\'s Law Quiz',
      subtitle: 'Generate a medium difficulty interactive MCQ',
      mode: 'quiz' as ModelMode,
      icon: HelpCircle,
      prompt: 'Generate a medium difficulty MCQ on Coulomb\'s law.',
      color: 'from-amber-500/20 to-orange-500/10 hover:border-amber-500/40',
      iconColor: 'text-amber-400',
    },
    {
      title: 'Diagnose Physics Mistake',
      subtitle: 'Analyze touched conducting spheres charge distribution',
      mode: 'diagnostic' as ModelMode,
      icon: SearchCheck,
      prompt: 'Diagnose: I answered F/4 instead of 3F/8 on two touched conducting spheres with unequal charges.',
      color: 'from-emerald-500/20 to-teal-500/10 hover:border-emerald-500/40',
      iconColor: 'text-emerald-400',
    },
    {
      title: 'Exam Anxiety Reset',
      subtitle: '1-minute breathing exercise and grounding support',
      mode: 'companion' as ModelMode,
      icon: HeartHandshake,
      prompt: 'I have my physics exam tomorrow and I\'m feeling completely overwhelmed. Can you guide me through a quick reset?',
      color: 'from-rose-500/20 to-purple-500/10 hover:border-rose-500/40',
      iconColor: 'text-rose-400',
    },
  ];

  return (
    <div className="max-w-3xl mx-auto px-4 py-12 flex flex-col items-center justify-center min-h-[60vh] text-center select-none">
      {/* Sparkle icon */}
      <div className="w-14 h-14 rounded-3xl bg-gradient-to-tr from-blue-500 via-indigo-500 to-rose-400 p-[2px] mb-6 shadow-2xl shadow-indigo-500/30 animate-pulse-subtle">
        <div className="w-full h-full bg-[#131314] rounded-3xl flex items-center justify-center">
          <Sparkles className="w-7 h-7 text-transparent bg-clip-text bg-gradient-to-tr from-blue-400 to-rose-300" />
        </div>
      </div>

      {/* Main Greeting */}
      <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mb-3">
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-300 to-rose-300 animate-gemini-gradient">
          Hello, Joshua
        </span>
      </h1>
      <p className="text-lg sm:text-xl text-[#c4c7c5] font-light max-w-lg mb-10 leading-relaxed">
        How can StudyVerse AI assist your learning journey today?
      </p>

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full text-left">
        {suggestions.map((card, idx) => {
          const Icon = card.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(card.prompt, card.mode)}
              className={`p-4 rounded-2xl bg-[#1e1f20] border border-[#333538] hover:bg-[#282a2c] transition-all duration-200 flex flex-col justify-between group shadow-sm hover:shadow-md hover:-translate-y-0.5 ${card.color}`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs uppercase tracking-wider font-semibold text-[#8e918f] group-hover:text-[#c4c7c5]">
                    {card.mode}
                  </span>
                  <div className="w-8 h-8 rounded-full bg-[#131314] flex items-center justify-center border border-[#333538]/50 group-hover:border-transparent">
                    <Icon className={`w-4 h-4 ${card.iconColor}`} />
                  </div>
                </div>
                <h3 className="text-sm font-medium text-[#e3e3e3] group-hover:text-white mb-1">
                  {card.title}
                </h3>
                <p className="text-xs text-[#8e918f] line-clamp-2">
                  {card.subtitle}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
