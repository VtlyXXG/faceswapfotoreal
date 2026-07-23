import { AIProviderError } from '../../utils/errors.js';
import { BaseTextProvider } from './BaseTextProvider.js';

/**
 * llama.cpp server в режиме OpenAI-совместимого API
 * (`llama-server --host 127.0.0.1 --port 8080`).
 */
export class LlamaCppTextProvider extends BaseTextProvider {
  constructor(options) {
    super(options);
    this.name = 'llamacpp';
  }

  async generate({ system, prompt, maxTokens, temperature, signal }) {
    const messages = [];
    if (system) messages.push({ role: 'system', content: system });
    messages.push({ role: 'user', content: prompt });

    let response;
    try {
      response = await fetch(new URL('/v1/chat/completions', this.options.baseUrl), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: this.options.model,
          messages,
          temperature: temperature ?? this.options.temperature,
          max_tokens: maxTokens ?? this.options.maxTokens,
          stream: false,
        }),
        signal: signal ?? AbortSignal.timeout(this.options.timeoutMs),
      });
    } catch (cause) {
      throw new AIProviderError('Сервер llama.cpp недоступен', { cause: cause.message });
    }

    if (!response.ok) {
      throw new AIProviderError('llama.cpp вернул ошибку', { status: response.status });
    }

    const data = await response.json();
    return {
      text: data.choices?.[0]?.message?.content ?? '',
      model: data.model ?? this.options.model,
      usage: {
        promptTokens: data.usage?.prompt_tokens,
        completionTokens: data.usage?.completion_tokens,
      },
    };
  }

  async healthCheck() {
    try {
      const response = await fetch(new URL('/health', this.options.baseUrl), {
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

export default LlamaCppTextProvider;
