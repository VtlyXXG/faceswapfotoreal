import { z } from 'zod';

/**
 * Спецификация заказа обложки — единственный источник правды для пайплайна.
 * Та же схема описывает и разовый вызов /personalize: работа одна и та же
 * (замена лица на обложке), различается только наличие доменного заказа.
 */
export const coverSpecSchema = z.object({
  // Человекочитаемый ярлык заказа — нужен только для списка и статуса
  title: z.string().min(1).max(200).optional(),

  // Пути к уже загруженным файлам под STORAGE_ROOT (server-side ссылки)
  source: z.string().min(1), // лицо, которое переносим
  target: z.string().min(1), // обложка, на которую переносим

  // Параметры ML-сервиса; имена совпадают с полями его формы
  options: z
    .object({
      enhance: z.boolean().optional(),
      style_strength: z.number().min(0).max(2).optional(),
      art_style: z.string().max(120).optional(),
      output_format: z.enum(['png', 'jpg']).optional(),
    })
    .strict()
    .default({}),
});

/** @typedef {z.infer<typeof coverSpecSchema>} CoverSpec */

export const parseCoverSpec = (input) => coverSpecSchema.parse(input);
