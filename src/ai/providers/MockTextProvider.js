import { BaseTextProvider } from './BaseTextProvider.js';

/** Детерминированная заглушка — позволяет гонять пайплайн без запущенной модели. */
export class MockTextProvider extends BaseTextProvider {
  constructor(options) {
    super(options);
    this.name = 'mock';
  }

  async generate({ prompt }) {
    return {
      text: `[mock] Сгенерированный фрагмент для запроса: ${prompt.slice(0, 120)}`,
      model: 'mock-text',
      usage: { promptTokens: 0, completionTokens: 0 },
    };
  }

  async healthCheck() {
    return true;
  }
}

export default MockTextProvider;
