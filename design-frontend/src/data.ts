export type Screen =
  | "login"
  | "personality"
  | "home"
  | "mode"
  | "subcategory"
  | "scenario"
  | "persona"
  | "live"
  | "evaluation"
  | "history"
  | "settings"
  | "admin";

export interface Persona {
  id: string;
  name: string;
  role: string;
  traits: string[];
  avatar: string;
  culture: "Pakistani" | "American";
}

export interface Session {
  id: string;
  date: string;
  mode: string;
  subcategory: string;
  persona: string;
  duration: string;
  score: number;
  skills: Record<string, number>;
}

export interface AppState {
  user: { name: string; email: string } | null;
  isFirstTime: boolean;
  selectedMode: string;
  selectedSubcategory: string;
  scenario: string;
  selectedPersona: Persona | null;
  difficulty: number;
  duration: number;
  culture: "Pakistani" | "American";
  sessions: Session[];
  personas: Persona[];
  personalityAnswers: Record<number, string | number>;
  skills: Record<string, number>;
  overallScore: number;
}

export const SUBCATEGORIES: Record<string, { label: string; icon: string }[]> = {
  Professional: [
    { label: "Interview", icon: "" },
    { label: "Difficult conversation with boss", icon: "" },
    { label: "Product pitch", icon: "" },
    { label: "Team conflict", icon: "" },
    { label: "Salary negotiation", icon: "" },
    { label: "Client meeting", icon: "" },
  ],
  Personal: [
    { label: "Talking to a stranger", icon: "" },
    { label: "Talking to a girl/guy", icon: "" },
    { label: "Family conflict", icon: "" },
    { label: "Making new friends", icon: "" },
    { label: "Networking event", icon: "" },
    { label: "Asking for help", icon: "" },
  ],
};

export const MOCK_PERSONAS: Persona[] = [
  {
    id: "1",
    name: "Sarah Chen",
    role: "Tech Hiring Manager",
    traits: ["Direct", "Detail-oriented", "Analytical"],
    avatar: "SC",
    culture: "American",
  },
  {
    id: "2",
    name: "Ahmed Raza",
    role: "Senior Colleague",
    traits: ["Warm", "Collaborative", "Experienced"],
    avatar: "AR",
    culture: "Pakistani",
  },
  {
    id: "3",
    name: "Maya Torres",
    role: "New Acquaintance",
    traits: ["Curious", "Friendly", "Open"],
    avatar: "MT",
    culture: "American",
  },
];

export const MOCK_SESSIONS: Session[] = [
  {
    id: "s1",
    date: "Aug 18, 2026",
    mode: "Professional",
    subcategory: "Interview",
    persona: "Sarah Chen",
    duration: "12 min",
    score: 78,
    skills: { Clarity: 80, Empathy: 72, Assertiveness: 85, "Active Listening": 70, Composure: 75, Adaptability: 68 },
  },
  {
    id: "s2",
    date: "Aug 16, 2026",
    mode: "Personal",
    subcategory: "Talking to a stranger",
    persona: "Maya Torres",
    duration: "8 min",
    score: 65,
    skills: { Clarity: 60, Empathy: 78, Assertiveness: 55, "Active Listening": 72, Composure: 68, Adaptability: 60 },
  },
  {
    id: "s3",
    date: "Aug 14, 2026",
    mode: "Professional",
    subcategory: "Salary negotiation",
    persona: "Ahmed Raza",
    duration: "15 min",
    score: 82,
    skills: { Clarity: 85, Empathy: 75, Assertiveness: 90, "Active Listening": 80, Composure: 82, Adaptability: 78 },
  },
];

export const SKILL_COLORS: Record<string, string> = {
  Clarity: "#326080",
  Empathy: "#805232",
  Assertiveness: "#4a90b8",
  "Active Listening": "#6b9e6b",
  Composure: "#9b7e6b",
  Adaptability: "#7a6b9b",
};

export const QUESTIONS = [
  {
    id: 1,
    type: "scale" as const,
    text: "When I need something from someone, I explain what I need and why it matters.",
  },
  {
    id: 2,
    type: "scale" as const,
    text: "When a conversation becomes tense, I can pause before I respond.",
  },
  {
    id: 3,
    type: "mcq" as const,
    text: "A friend cancels plans at the last minute for the second time. What do you usually say first?",
    options: [
      "I understand things come up. Let me know when you have more space.",
      "I was looking forward to this. Can we talk about what keeps changing?",
      '"No problem," then wait to see whether they suggest another time.',
      "I am disappointed. If plans may change, please tell me earlier.",
    ],
  },
  {
    id: 4,
    type: "mcq" as const,
    text: "A family member shares a problem but seems unsure what they want from you. What would you usually do first?",
    options: [
      "Offer a practical solution based on what has helped you before.",
      "Ask whether they want advice, listening, or help with a next step.",
      "Reassure them and share a similar experience from your own life.",
      "Give them time to explain, then summarize what you heard.",
    ],
  },
  {
    id: 5,
    type: "mcq" as const,
    text: "Your manager asks for a task by tomorrow, but you already have two urgent commitments. Which response is most like you?",
    options: [
      "Accept it and reorganize your work later if the deadline becomes difficult.",
      "Explain the conflict, name what can move, and ask which priority comes first.",
      "Say tomorrow is unrealistic and suggest a later date without discussing tradeoffs.",
      "Ask a teammate to help before telling your manager there is a conflict.",
    ],
  },
  {
    id: 6,
    type: "mcq" as const,
    text: "You are explaining a complicated idea to someone unfamiliar with the topic. Which approach best describes you?",
    options: [
      "Start with the main point, use familiar words, and check what they need next.",
      "Give the full background first so the explanation is complete.",
      "Use an analogy and adjust the detail after hearing their questions.",
      "Ask what they already know, then tailor the explanation from there.",
    ],
  },
  {
    id: 7,
    type: "mcq" as const,
    text: "In a group discussion, someone disagrees with your suggestion and gives a reason you had not considered. What would you usually do first?",
    options: [
      "Explain why your original idea still solves the main problem.",
      "Ask one question to understand their concern before responding.",
      "Suggest a small trial so both ideas can be compared.",
      "Set your idea aside and invite them to develop their alternative.",
    ],
  },
  {
    id: 8,
    type: "mcq" as const,
    text: "You notice your voice becoming sharper during a difficult conversation. When this happens, which reaction is closest to yours?",
    options: [
      "Slow down, name the tension, and continue with the specific issue.",
      "Keep the conversation brief and return to it after you have cooled off.",
      "Focus on the facts and avoid discussing how either person feels.",
      "Ask for a short pause, then come back with a clear next step.",
    ],
  },
  {
    id: 9,
    type: "mcq" as const,
    text: "A teammate misses an agreed handoff, and your work is now delayed. What would you usually do first?",
    options: [
      "Fix the immediate problem yourself and mention the delay later.",
      "Send a reminder to the whole group about the agreed timing.",
      "Speak privately about the impact, ask what happened, and agree on a reliable next step.",
      "Tell them the handoff needs to happen today and copy the team lead.",
    ],
  },
  {
    id: 10,
    type: "mcq" as const,
    text: "Someone gives you feedback that your usual communication style is hard for them to follow. Which response is most like you?",
    options: [
      "Explain your intention and continue using the same approach for consistency.",
      "Ask what was difficult, try a different approach, and check whether it helped.",
      "Give them control over when and how you communicate in future.",
      "Reduce the amount of feedback until you understand their preferences better.",
    ],
  },
];
