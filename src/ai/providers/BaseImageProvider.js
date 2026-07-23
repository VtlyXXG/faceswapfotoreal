/**
 * Контракт провайдера изображений (Automatic1111, ComfyUI и т.п.).
 */
export class BaseImageProvider {
  constructor(options) {
    this.options = options;
    this.name = 'base';
  }

  /**
   * @param {{ prompt: string, negativePrompt?: string, width?: number, height?: number, seed?: number, signal?: AbortSignal }} _params
   * @returns {Promise<{ images: Array<{ data: Buffer, mimeType: string }>, model: string }>}
   */
  async generate(_params) {
    throw new Error(`${this.name}: метод generate() не реализован`);
  }

  async healthCheck() {
    return false;
  }
}

export default BaseImageProvider;
