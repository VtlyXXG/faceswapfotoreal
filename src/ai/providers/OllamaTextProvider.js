import { AIProviderError } from '../../utils/errors.js';
import { BaseTextProvider } from './BaseTextProvider.js';

/** Локальный Ollama-сервер (http://127.0.0.1:11434). */
export class OllamaTextProvider extends BaseTextProvider {
  constructor(options) {
    super(options);
    this.name = 'ollama';
  }

  #request(path, body, signal) {
    return fetch(new URL(path, this.options.baseUrl), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: signal ?? AbortSignal.timeout(this.options.timeoutMs),
    });
  }

  #payload({ system, prompt, maxTokens, temperature, stream }) {
    return {
      model: this.options.model,
      prompt,
      system,
      stream,
      options: {
        temperature: temperature ?? this.options.temperature,
        num_predict: maxTokens ?? this.options.maxTokens,
      },
    };
  }

  async generate({ system, prompt, maxTokens, temperature, signal }) {
    let response;
    try {
      response = await this.#request(
        '/api/generate',
        this.#payload({ system, prompt, maxTokens, temperature, stream: false }),
        signal,
      );
    } catch (cause) {
      throw new AIProviderError('Локальная модель Ollama недоступна', { cause: cause.message });
    }

    if (!response.ok) {
      throw new AIProviderError('Ollama вернул ошибку', {
        status: response.status,
        body: await response.text().catch(() => ''),
      });
    }

    const data = await response.json();
    return {
      text: data.response ?? '',
      model: data.model ?? this.options.model,
      usage: {
        promptTokens: data.prompt_eval_count,
        completionTokens: data.eval_count,
      },
    };
  }

  async *stream({ system, prompt, maxTokens, temperature, signal }) {
    const response = await this.#request(
      '/api/generate',
      this.#payload({ system, prompt, maxTokens, temperature, stream: true }),
      signal,
    );

    if (!response.ok || !response.body) {
      throw new AIProviderError('Ollama не открыл поток', { status: response.status });
    }

    const decoder = new TextDecoder();
    let buffer = '';

    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.trim()) continue;
        const parsed = JSON.parse(line);
        if (parsed.response) yield parsed.response;
        if (parsed.done) return;
      }
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(new URL('/api/tags', this.options.baseUrl), {
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

export default OllamaTextProvider;
