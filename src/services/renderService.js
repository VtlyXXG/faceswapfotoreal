/**
 * Сборка готовых файлов книги из сгенерированных глав.
 * PDF/EPUB пока не реализованы — рендереры подключаются сюда без изменения пайплайна.
 */

const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch],
  );

const renderMarkdown = (book) =>
  [
    `# ${book.title}`,
    '',
    ...book.chapters.flatMap((c) => [
      `## Глава ${c.index}. ${c.title}`,
      '',
      c.illustration ? `![Иллюстрация к главе ${c.index}](${c.illustration.filename})` : null,
      c.illustration ? '' : null,
      c.content,
      '',
    ]),
  ]
    .filter((line) => line !== null)
    .join('\n');

const renderHtml = (book) => `<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>${escapeHtml(book.title)}</title>
<style>
  body { font-family: Georgia, serif; max-width: 42rem; margin: 3rem auto; line-height: 1.6; padding: 0 1rem; }
  img { max-width: 100%; }
</style>
</head>
<body>
<h1>${escapeHtml(book.title)}</h1>
${book.chapters
  .map(
    (c) => `<section>
  <h2>Глава ${c.index}. ${escapeHtml(c.title)}</h2>
  ${c.illustration ? `<img src="${escapeHtml(c.illustration.filename)}" alt="">` : ''}
  ${c.content
    .split(/\n{2,}/)
    .map((p) => `<p>${escapeHtml(p)}</p>`)
    .join('\n  ')}
</section>`,
  )
  .join('\n')}
</body>
</html>`;

const RENDERERS = {
  md: { render: renderMarkdown, filename: 'book.md' },
  html: { render: renderHtml, filename: 'book.html' },
};

export const renderBook = (book, format) => {
  const renderer = RENDERERS[format];
  if (!renderer) {
    // TODO: pdf (puppeteer / pdfkit), epub (epub-gen)
    return null;
  }
  return { filename: renderer.filename, content: renderer.render(book) };
};

export const supportedFormats = () => Object.keys(RENDERERS);
