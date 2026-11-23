export type ModelMode = 'tutor' | 'quiz' | 'diagnostic' | 'companion' | 'general';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  mode?: ModelMode;
  status?: 'pending' | 'streaming' | 'complete' | 'error';
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  mode: ModelMode;
}

export interface AppSettings {
  apiUrl: string;
  model: string;
  temperature: number;
  maxTokens: number;
}

export interface ModeConfig {
  id: ModelMode;
  name: string;
  label: string;
  badge: string;
  description: string;
  icon: string;
  systemPrompt: string;
  gradient: string;
}

export const MODES: Record<ModelMode, ModeConfig> = {
  general: {
    id: 'general',
    name: 'General Assistant',
    label: 'General',
    badge: 'Standard',
    description: 'Versatile academic and problem-solving assistance',
    icon: 'Sparkles',
    systemPrompt: 'You are StudyVerse AI, a helpful, intelligent academic AI assistant based on fine-tuned Qwen 2.5 3B.',
    gradient: 'from-blue-400 via-indigo-400 to-purple-400',
  },
  tutor: {
    id: 'tutor',
    name: 'Academic Tutor',
    label: 'Tutor',
    badge: '[Mode: tutor]',
    description: 'Socratic step-by-step explanations with formatted LaTeX math ($...$)',
    icon: 'GraduationCap',
    systemPrompt: 'You are StudyVerse AI tutor. [Mode: tutor] Provide concise, syllabus-grounded academic explanations with clear LaTeX math equations.',
    gradient: 'from-indigo-400 via-blue-400 to-cyan-400',
  },
  quiz: {
    id: 'quiz',
    name: 'Adaptive Quiz',
    label: 'Quiz',
    badge: '[Mode: quiz]',
    description: 'Generates structured practice questions and conceptual tests',
    icon: 'HelpCircle',
    systemPrompt: 'You are StudyVerse AI quiz generator. [Mode: quiz] Provide challenging, pedagogical practice questions.',
    gradient: 'from-amber-400 via-orange-400 to-rose-400',
  },
  diagnostic: {
    id: 'diagnostic',
    name: 'Mistake Diagnostic',
    label: 'Diagnostic',
    badge: '[Mode: diagnostic]',
    description: 'Identifies root-cause error traps and provides remediation steps',
    icon: 'SearchCheck',
    systemPrompt: 'You are StudyVerse AI diagnostic assistant. [Mode: diagnostic] Diagnose the root cause of academic mistakes and provide repair cards.',
    gradient: 'from-emerald-400 via-teal-400 to-cyan-400',
  },
  companion: {
    id: 'companion',
    name: 'Calm Companion',
    label: 'Companion',
    badge: '[Mode: companion]',
    description: 'Empathetic exam stress relief, grounding resets, and micro-tasks',
    icon: 'HeartHandshake',
    systemPrompt: 'You are StudyVerse AI calm companion. [Mode: companion] Provide empathetic, grounded emotional support with zero toxic positivity and 1-minute breathing resets.',
    gradient: 'from-rose-400 via-pink-400 to-purple-400',
  },
};
