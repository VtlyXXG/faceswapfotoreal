import { BaseImageProvider } from './BaseImageProvider.js';

/** 1x1 PNG-заглушка для работы без запущенной модели изображений. */
const PLACEHOLDER_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
  'base64',
);

export class MockImageProvider extends BaseImageProvider {
  constructor(options) {
    super(options);
    this.name = 'mock';
  }

  async generate() {
    return {
      images: [{ data: PLACEHOLDER_PNG, mimeType: 'image/png' }],
      model: 'mock-image',
    };
  }

  async healthCheck() {
    return true;
  }
}

export default MockImageProvider;
