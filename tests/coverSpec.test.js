import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { parseCoverSpec } from '../src/domain/coverSpec.js';

describe('coverSpec', () => {
  test('подставляет значения по умолчанию', () => {
    const spec = parseCoverSpec({ source: 'uploads/face.jpg', target: 'uploads/cover.png' });

    assert.deepEqual(spec.options, {});
    assert.equal(spec.title, undefined);
  });

  test('отклоняет заказ без исходников', () => {
    assert.throws(() => parseCoverSpec({ source: 'uploads/face.jpg' }));
    assert.throws(() => parseCoverSpec({ target: 'uploads/cover.png' }));
  });

  test('отклоняет неизвестный параметр ML-сервиса', () => {
    assert.throws(() =>
      parseCoverSpec({
        source: 'uploads/face.jpg',
        target: 'uploads/cover.png',
        options: { unknown: 1 },
      }),
    );
  });

  test('отклоняет силу стилизации вне диапазона', () => {
    assert.throws(() =>
      parseCoverSpec({
        source: 'uploads/face.jpg',
        target: 'uploads/cover.png',
        options: { style_strength: 5 },
      }),
    );
  });
});
