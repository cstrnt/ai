import { Mistral } from '@mistralai/mistralai'

export interface MistralClientConfig {
  /** Mistral API key. */
  apiKey: string

  /** Optional server URL override. */
  serverURL?: string

  /** Optional request timeout (ms). */
  timeoutMs?: number
}

/**
 * Creates a Mistral SDK client instance.
 */
export function createMistralClient(config: MistralClientConfig): Mistral {
  const { apiKey, serverURL, timeoutMs } = config
  return new Mistral({
    apiKey,
    ...(serverURL ? { serverURL } : {}),
    ...(timeoutMs ? { timeoutMs } : {}),
  })
}

/**
 * Gets Mistral API key from environment variables.
 * @throws Error if MISTRAL_API_KEY is not found
 */
export function getMistralApiKeyFromEnv(): string {
  const env =
    typeof globalThis !== 'undefined' && (globalThis as any).window?.env
      ? (globalThis as any).window.env
      : typeof process !== 'undefined'
        ? process.env
        : undefined
  const key = env?.MISTRAL_API_KEY

  if (!key) {
    throw new Error(
      'MISTRAL_API_KEY is required. Please set it in your environment variables or use the factory function with an explicit API key.',
    )
  }

  return key
}

/**
 * Generates a unique ID with a prefix.
 */
export function generateId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substring(7)}`
}
