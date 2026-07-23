import { getTextProvider, getImageProvider } from '../ai/registry.js';
import {
  SYSTEM_AUTHOR,
  outlinePrompt,
  chapterPrompt,
  illustrationPrompt,
  NEGATIVE_ILLUSTRATION_PROMPT,
} from '../ai/prompts/bookPrompts.js';
import { createChapter } from '../domain/book.js';
import { AIProviderError } from '../utils/errors.js';
import { createLogger } from '../utils/logger.js';

const log = createLogger('generation');

/** Достаёт JSON из ответа модели, даже если он обёрнут в ```json ... ```. */
const extractJson = (raw) => {
  const cleaned = raw.replace(/```(?:json)?/gi, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end === -1) {
    throw new AIProviderError('Модель не вернула JSON-структуру', { raw: raw.slice(0, 500) });
  }
  return JSON.parse(cleaned.slice(start, end + 1));
};

/** Шаг 1 — структура книги. */
export const generateOutline = async (spec) => {
  const provider = getTextProvider();
  const { text } = await provider.generate({
    system: SYSTEM_AUTHOR,
    prompt: outlinePrompt(spec),
    temperature: 0.7,
  });

  let parsed;
  try {
    parsed = extractJson(text);
  } catch (err) {
    log.warn({ err }, 'не удалось разобрать структуру, используется запасной вариант');
    parsed = {
      title: spec.title ?? 'Персональная книга',
      chapters: Array.from({ length: spec.chapterCount }, (_, i) => ({
        index: i + 1,
        title: `Глава ${i + 1}`,
        summary: '',
      })),
    };
  }

  return {
    title: parsed.title ?? spec.title ?? 'Персональная книга',
    chapters: (parsed.chapters ?? [])
      .slice(0, spec.chapterCount)
      .map((c, i) => createChapter({ index: c.index ?? i + 1, title: c.title, summary: c.summary })),
  };
};

/** Шаг 2 — текст одной главы (с учётом предыдущего контекста). */
export const generateChapter = async (spec, chapter, previousSummary) => {
  const provider = getTextProvider();
  const { text } = await provider.generate({
    system: SYSTEM_AUTHOR,
    prompt: chapterPrompt(spec, chapter, previousSummary),
  });
  return { ...chapter, content: text.trim() };
};

/** Шаг 3 — иллюстрация к главе (опционально). */
export const generateIllustration = async (spec, chapter) => {
  const provider = getImageProvider();
  const { images, model } = await provider.generate({
    prompt: illustrationPrompt(spec, chapter),
    negativePrompt: NEGATIVE_ILLUSTRATION_PROMPT,
  });
  const image = images[0];
  return image ? { data: image.data, mimeType: image.mimeType, model } : null;
};
