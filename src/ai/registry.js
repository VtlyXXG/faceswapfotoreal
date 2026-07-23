import { config } from '../config/index.js';
import { AppError } from '../utils/errors.js';
import { createLogger } from '../utils/logger.js';
import { OllamaTextProvider } from './providers/OllamaTextProvider.js';
import { LlamaCppTextProvider } from './providers/LlamaCppTextProvider.js';
import { MockTextProvider } from './providers/MockTextProvider.js';
import { Automatic1111ImageProvider } from './providers/Automatic1111ImageProvider.js';
import { MockImageProvider } from './providers/MockImageProvider.js';

const log = createLogger('ai:registry');

const TEXT_PROVIDERS = {
  ollama: OllamaTextProvider,
  llamacpp: LlamaCppTextProvider,
  mock: MockTextProvider,
};

const IMAGE_PROVIDERS = {
  automatic1111: Automatic1111ImageProvider,
  comfyui: Automatic1111ImageProvider, // TODO: отдельная реализация под ComfyUI graph API
  mock: MockImageProvider,
};

let textProvider;
let imageProvider;

const instantiate = (table, name, options, kind) => {
  const Provider = table[name];
  if (!Provider) {
    throw new AppError(
      `Неизвестный ${kind}-провайдер "${name}". Доступны: ${Object.keys(table).join(', ')}`,
      { code: 'UNKNOWN_AI_PROVIDER' },
    );
  }
  log.info({ kind, provider: name, model: options.model }, 'ИИ-провайдер инициализирован');
  return new Provider(options);
};

export const getTextProvider = () => {
  textProvider ??= instantiate(TEXT_PROVIDERS, config.ai.text.provider, config.ai.text, 'text');
  return textProvider;
};

export const getImageProvider = () => {
  imageProvider ??= instantiate(IMAGE_PROVIDERS, config.ai.image.provider, config.ai.image, 'image');
  return imageProvider;
};

/** Регистрация собственной реализации — точка расширения для новых локальных моделей. */
export const registerTextProvider = (name, Provider) => {
  TEXT_PROVIDERS[name] = Provider;
};

export const registerImageProvider = (name, Provider) => {
  IMAGE_PROVIDERS[name] = Provider;
};

/** Сброс синглтонов (нужен тестам и горячей смене конфигурации). */
export const resetProviders = () => {
  textProvider = undefined;
  imageProvider = undefined;
};

export const checkProviders = async () => {
  const [text, image] = await Promise.all([
    getTextProvider().healthCheck(),
    getImageProvider().healthCheck(),
  ]);
  return {
    text: { provider: config.ai.text.provider, model: config.ai.text.model, available: text },
    image: { provider: config.ai.image.provider, model: config.ai.image.model, available: image },
  };
};
