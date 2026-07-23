import { z } from 'zod';

/** Спецификация заказа книги — единственный источник правды для пайплайна генерации. */
export const bookSpecSchema = z.object({
  title: z.string().min(1).max(200).optional(),
  chapterCount: z.number().int().min(1).max(50).default(5),
  wordsPerChapter: z.number().int().min(100).max(5000).default(700),
  withIllustrations: z.boolean().default(false),

  recipient: z
    .object({
      name: z.string().min(1).max(100),
      age: z.number().int().min(0).max(120).optional(),
      gender: z.enum(['male', 'female', 'other']).optional(),
      interests: z.array(z.string().min(1).max(50)).max(20).default([]),
    })
    .strict(),

  style: z
    .object({
      genre: z.string().max(80).default('приключения'),
      tone: z.string().max(80).default('тёплый и добрый'),
      language: z.string().max(20).default('ru'),
      artStyle: z.string().max(80).optional(),
    })
    .strict()
    .default({}),

  output: z
    .object({
      formats: z.array(z.enum(['md', 'html', 'pdf', 'epub'])).min(1).default(['md']),
    })
    .strict()
    .default({}),
});

/** @typedef {z.infer<typeof bookSpecSchema>} BookSpec */

export const parseBookSpec = (input) => bookSpecSchema.parse(input);
