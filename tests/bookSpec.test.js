import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { parseBookSpec } from '../src/domain/bookSpec.js';

describe('bookSpec', () => {
  test('подставляет значения по умолчанию', () => {
    const spec = parseBookSpec({ recipient: { name: 'Аня' } });

    assert.equal(spec.chapterCount, 5);
    assert.equal(spec.withIllustrations, false);
    assert.deepEqual(spec.output.formats, ['md']);
    assert.equal(spec.style.language, 'ru');
  });

  test('отклоняет заказ без получателя', () => {
    assert.throws(() => parseBookSpec({ chapterCount: 3 }));
  });

  test('отклоняет слишком большое число глав', () => {
    assert.throws(() => parseBookSpec({ recipient: { name: 'Аня' }, chapterCount: 500 }));
  });
});
