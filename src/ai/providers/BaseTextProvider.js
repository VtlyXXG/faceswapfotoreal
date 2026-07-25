/**
 * Контракт текстового провайдера.
 * Любая локальная модель (llama.cpp, vLLM) реализует этот интерфейс,
 * поэтому доменный код никогда не знает, какая модель стоит за вызовом.
 */
export class BaseTextProvider {
  /** @param {import('../../config/index.js').config['ai']['text']} options */
  constructor(options) {
    this.options = options;
    this.name = 'base';
  }

  /**
   * Однократная генерация текста.
   * @param {{ system?: string, prompt: string, maxTokens?: number, temperature?: number, signal?: AbortSignal }} _params
   * @returns {Promise<{ text: string, model: string, usage?: object }>}
   */
  async generate(_params) {
    throw new Error(`${this.name}: метод generate() не реализован`);
  }

  /**
   * Потоковая генерация — по умолчанию сводится к generate().
   * @param {object} params
   * @returns {AsyncGenerator<string>}
   */
  async *stream(params) {
    const { text } = await this.generate(params);
    yield text;
  }

  /** Проверка доступности локальной модели. @returns {Promise<boolean>} */
  async healthCheck() {
    return false;
  }
}

export default BaseTextProvider;
