/**
 * Шаблоны промптов для пайплайна генерации книги.
 * Держим их отдельно от бизнес-логики: промпт — это версионируемый артефакт.
 */

export const SYSTEM_AUTHOR = `Ты — профессиональный автор персонализированных книг.
Пиши живым языком, соблюдай единый стиль, возрастную аудиторию и заданные параметры персонажа.
Не добавляй метакомментариев и служебных пояснений — только текст произведения.`;

const personaBlock = ({ recipient = {}, style = {} }) =>
  [
    recipient.name && `Главный герой: ${recipient.name}`,
    recipient.age && `Возраст читателя: ${recipient.age}`,
    recipient.gender && `Пол героя: ${recipient.gender}`,
    recipient.interests?.length && `Интересы: ${recipient.interests.join(', ')}`,
    style.tone && `Тон повествования: ${style.tone}`,
    style.genre && `Жанр: ${style.genre}`,
    style.language && `Язык: ${style.language}`,
  ]
    .filter(Boolean)
    .join('\n');

/** Промпт для построения оглавления/структуры книги. */
export const outlinePrompt = (spec) => `${personaBlock(spec)}

Тема книги: ${spec.title ?? 'на твоё усмотрение'}
Количество глав: ${spec.chapterCount}

Составь структуру книги. Верни строго JSON без markdown-обёртки:
{"title": "...", "chapters": [{"index": 1, "title": "...", "summary": "..."}]}`;

/** Промпт для написания одной главы. */
export const chapterPrompt = (spec, chapter, previousSummary) => `${personaBlock(spec)}

Книга: «${spec.title}»
Глава ${chapter.index}: «${chapter.title}»
Краткое содержание главы: ${chapter.summary}
${previousSummary ? `Что произошло ранее: ${previousSummary}` : ''}

Напиши полный текст главы объёмом примерно ${spec.wordsPerChapter ?? 700} слов.`;

/** Промпт для иллюстрации к главе. */
export const illustrationPrompt = (spec, chapter) =>
  [
    `book illustration, ${spec.style?.artStyle ?? 'soft watercolor'}`,
    chapter.title,
    chapter.summary,
    spec.recipient?.name ? `main character ${spec.recipient.name}` : null,
    'high detail, coherent composition',
  ]
    .filter(Boolean)
    .join(', ');

export const NEGATIVE_ILLUSTRATION_PROMPT =
  'text, watermark, signature, extra limbs, deformed hands, low quality, blurry';
