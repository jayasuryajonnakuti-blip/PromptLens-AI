export interface ExamplePrompt {
  id: string
  title: string
  category: string
  prompt: string
}

export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    id: '1',
    title: 'Executive Summary',
    category: 'Business',
    prompt: 'Summarize this quarterly earnings report for executive leadership. Highlight key revenue growth drivers and risk factors in under 300 words.',
  },
  {
    id: '2',
    title: 'Python CLI Tool',
    category: 'Coding',
    prompt: 'Write a Python CLI tool using argparse that monitors folder file changes and logs them into a JSON lines file.',
  },
  {
    id: '3',
    title: 'Concept Explanation',
    category: 'Education',
    prompt: 'Explain how neural network backpropagation works to a computer science undergraduate student without using heavy calculus notation.',
  },
  {
    id: '4',
    title: 'Marketing Launch Email',
    category: 'Marketing',
    prompt: 'Draft an email announcing a new AI feature to existing enterprise SaaS users. Focus on productivity benefits and include a clear call-to-action button text.',
  },
]
