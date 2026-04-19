import type { Provider, Feature } from '../src/lib/types'
import { isSupported } from '../src/lib/feature-support'

export { isSupported }

export const providers: Provider[] = [
  'openai',
  'anthropic',
  'gemini',
  'ollama',
  'groq',
  'grok',
  'openrouter',
  'mistral',
]

/** Get only the providers that support a given feature */
export function providersFor(feature: Feature): Provider[] {
  return providers.filter((p) => isSupported(p, feature))
}
