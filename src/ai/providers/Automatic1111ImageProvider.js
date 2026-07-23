import { AIProviderError } from '../../utils/errors.js';
import { BaseImageProvider } from './BaseImageProvider.js';

/** Локальный Stable Diffusion WebUI с включённым `--api`. */
export class Automatic1111ImageProvider extends BaseImageProvider {
  constructor(options) {
    super(options);
    this.name = 'automatic1111';
  }

  async generate({ prompt, negativePrompt, width, height, seed, signal }) {
    let response;
    try {
      response = await fetch(new URL('/sdapi/v1/txt2img', this.options.baseUrl), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          prompt,
          negative_prompt: negativePrompt ?? '',
          width: width ?? this.options.width,
          height: height ?? this.options.height,
          seed: seed ?? -1,
          steps: 30,
        }),
        signal: signal ?? AbortSignal.timeout(this.options.timeoutMs),
      });
    } catch (cause) {
      throw new AIProviderError('Локальный сервер изображений недоступен', {
        cause: cause.message,
      });
    }

    if (!response.ok) {
      throw new AIProviderError('Сервер изображений вернул ошибку', { status: response.status });
    }

    const data = await response.json();
    return {
      images: (data.images ?? []).map((base64) => ({
        data: Buffer.from(base64, 'base64'),
        mimeType: 'image/png',
      })),
      model: this.options.model,
    };
  }

  async healthCheck() {
    try {
      const response = await fetch(new URL('/sdapi/v1/options', this.options.baseUrl), {
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

export default Automatic1111ImageProvider;
