import { randomUUID } from 'node:crypto';

export const newId = (prefix) => (prefix ? `${prefix}_${randomUUID()}` : randomUUID());

export const slugify = (value) =>
  String(value)
    .toLowerCase()
    .replace(/[^a-z0-9а-яё]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
