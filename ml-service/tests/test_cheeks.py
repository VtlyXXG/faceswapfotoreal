"""
Разрушение структуры старой пряди в полосе вдоль щеки.

Проверяется главное свойство и четыре ловушки, каждая из которых стоила бы
живого прогона.

Свойство: под полосой не остаётся СТРУКТУРЫ — контрастной вытянутой линии, за
которую редактор цепляется, принимая её за тень скулы.

Ловушки. Тон нельзя мерить внутри полосы, иначе он придёт от самой пряди и мы
зальём её же цветом. Стороны различаются знаком проекции на ось лица, а не
порядком в коде, — иначе на склонённой голове тона меняются местами молча.
Заливка не должна ложиться на фон: пятно чужого тона внутри маски переживает
вклейку. И вне маски не меняется ни пикселя — на этом держится обещание, что
испорченные пиксели не попадут в готовый разворот.
"""

import numpy as np

from app.pipelines import cheeks

_FACE_HEIGHT = 120.0

# Пропорции сцены взяты не с потолка, а из `hair_mask`: полоса идёт от 0.75 до
# 1.8 полуширины лица, а центроид опорных точек щеки лежит около 0.70 — то есть
# на считанные пиксели МЕДИАЛЬНЕЕ внутренней кромки полосы. Замерено на живых
# кадрах, и в этом весь смысл сцены: диск сэмплирования обязан частично попадать
# на полосу, иначе главная ловушка модуля не проверяется вовсе
_AXIS = 160.0
_HALF = 60.0
_INNER, _OUTER, _CENTROID = 0.75, 1.8, 0.70


class _Parsed:
    """Разметка: класс лица, кожи тела, волос и одежды."""

    def __init__(self, face, skin, hair, clothes):
        self.face, self.skin, self.hair, self.clothes = face, skin, hair, clothes

    @property
    def bare_skin(self):
        return np.maximum(self.face, self.skin)


class _Hair:
    """`hair_mask.HairMask` ровно в тех полях, которые читает модуль."""

    def __init__(self, mask, cheeks_band, parsed, points, geometry):
        self.mask, self.cheeks, self.parsed = mask, cheeks_band, parsed
        self.points, self.geometry = points, geometry
        self.face_height = _FACE_HEIGHT


# Тона сцены разведены нарочно и далеко друг от друга: фон светлый, кожа
# средняя, прядь тёмная. Только так видно, ЧЕМ залит пиксель, — на близких
# тонах заливка фоном и заливка кожей дают почти одно и то же, и главный тест
# двухтоновой заливки не отличил бы их от размытия.
_SKY, _SKIN, _STRAND = 210, 150, 25

# Прядь над фоном: она и есть та, что раньше оставалась низкочастотным тёмным
# пятном. Лежит внутри левой полосы, но СНАРУЖИ лица — то есть ровно в зоне,
# которую прежняя версия оставляла на одном размытии.
_GHOST = (slice(150, 220), slice(70, 82))
# Прядь на щеке, внутри лица — контроль, что первая зона работает как работала.
_ON_FACE = (slice(150, 220), slice(104, 114))


def _scene(tone_left: int = _SKIN, tone_right: int = _SKIN):
    """
    Лицо на светлом фоне, две тёмные пряди: одна на щеке, вторая над фоном.

    Ось головы вертикальна, подбородок внизу: полосы ложатся слева и справа от
    лица, как на настоящем шаблоне.
    """
    axis = int(_AXIS)
    inner, outer = int(_HALF * _INNER), int(_HALF * _OUTER)

    image = np.full((320, 320, 3), _SKY, dtype=np.uint8)
    image[120:240, axis - 60 : axis] = tone_left
    image[120:240, axis : axis + 60] = tone_right
    image[_ON_FACE] = _STRAND
    image[_GHOST] = _STRAND

    face = np.zeros((320, 320), dtype=np.uint8)
    face[120:240, axis - 60 : axis + 60] = 255
    # Обе пряди разметка не относит НИ К ЧЕМУ: на лице это дыра в классе FACE,
    # над фоном — просто фон. Так себя и ведёт живой сегментатор
    face[_ON_FACE] = 0
    zeros = np.zeros((320, 320), dtype=np.uint8)
    parsed = _Parsed(face=face, skin=zeros.copy(), hair=zeros.copy(), clothes=zeros.copy())

    band = np.zeros((320, 320), dtype=np.uint8)
    band[102:282, axis - outer : axis - inner] = 255
    band[102:282, axis + inner : axis + outer] = 255

    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[90:290, 40:280] = 255

    offset = int(_HALF * _CENTROID)
    points = [(0, 0)] * 468
    # Подбородок и зеркальная пара углов челюсти: по ним строится линия, ниже
    # которой заливка тоном КОЖИ не идёт (шея освещена иначе)
    points[152] = (axis, 240)
    points[150], points[379] = (axis - 40, 230), (axis + 40, 230)
    for index, shift in ((50, 0), (117, -2), (205, 4)):
        points[index] = (axis - offset + shift, 180)
    for index, shift in ((280, 0), (346, 2), (425, -4)):
        points[index] = (axis + offset + shift, 180)

    geometry = {
        "chin": np.array([_AXIS, 240.0]),
        "up": np.array([0.0, -1.0]),
        "side": np.array([1.0, 0.0]),
        "face_height": _FACE_HEIGHT,
        "face_width": _HALF * 2,
    }
    return image, _Hair(mask, band, parsed, points, geometry)


def test_the_strand_loses_its_structure():
    """
    Главное. От пряди не остаётся перепада, за который можно зацепиться:
    падение лапласиана и есть «сохранять редактору нечего».
    """
    image, hair = _scene()

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["flattened"] is True
    assert meta["structure_after"] < meta["structure_before"] / 5

    # Меряется НУТРО пряди, без пары пикселей у внутренней кромки полосы: там
    # альфа растушёвана намеренно, и исходный тон проступает сквозь неё по
    # построению. Порог берётся от тонов сцены, а не константой — иначе он
    # молча устареет при первой же смене палитры теста
    core = float(np.min(flat[150:220, 104:111]))
    assert core > (_SKIN + _STRAND) / 2, "тёмные пиксели пряди остались"


def test_the_tone_is_never_measured_inside_the_band():
    """
    Ловушка, которая замыкается сама на себя. Прядь лежит внутри полосы; если
    сэмпл её прочитает, медиана уедет в тёмное, полоса зальётся тёмным — и мы
    получим ровно тот артефакт, который удаляли, не заметив этого ничем.
    """
    image, hair = _scene()

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert min(meta["tone_positive"]) > 120, "тон пришёл от пряди, а не от кожи"
    assert min(meta["tone_negative"]) > 120


def test_the_sides_are_told_apart_by_geometry_not_by_order():
    """
    Тон правой щеки не должен оказаться на левой полосе. Проверяется на кадре,
    где щёки заведомо разного тона: при совпадающих тонах подмена не видна
    вовсе — и потому в проде вылезла бы только на боковом свете.
    """
    image, hair = _scene(tone_left=120, tone_right=230)

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    # side = +x, значит правая щека кадра — положительная сторона
    assert meta["tone_positive"][0] > meta["tone_negative"][0]
    assert meta["sides_collapsed"] is False


def test_the_skin_tone_does_not_spill_onto_the_background():
    """
    Тон кожи по фону стал бы пятном чужого тона ВНУТРИ маски, то есть ореолом,
    пережившим вклейку. Зоны две, и каждая заливается СВОИМ тоном: проверяется
    не «фон не залит», а «фон залит фоном».
    """
    image, hair = _scene()

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    sky = float(np.mean(flat[110, 60]))
    assert abs(sky - _SKY) < abs(sky - _SKIN), "фон принял тон кожи"
    assert meta["filled_skin_px"] > 0 and meta["filled_background_px"] > 0


def test_the_ghost_over_the_background_is_overwritten_not_blurred():
    """
    Регрессия, стоившая композита. Прежняя версия заливала только кожу, а всё
    остальное отдавала размытию — и тёмная прядь над фоном не исчезала, а
    превращалась в мягкое тёмное пятно. Редактор читал его как тень заднего
    плана, то есть как часть сцены, которую положено сохранить.

    Проверяется физическая перезапись: пиксели пряди обязаны подтянуться к
    ФОНУ, а не остаться ближе к своей исходной темноте.
    """
    image, hair = _scene()

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    ghost = float(np.mean(flat[_GHOST]))
    assert abs(ghost - _SKY) < abs(ghost - _STRAND), "прядь над фоном осталась тёмной"
    assert ghost > (_SKY + _STRAND) / 2, "перезаписи не случилось, это всё ещё размытие"
    assert meta["filled_background_px"] > 0


def test_both_strands_die_but_by_different_tones():
    """
    Двухтоновость по существу: прядь на щеке уходит в тон кожи, прядь над
    фоном — в тон фона. Один тон на обе означал бы либо кожу на небе, либо
    небо на щеке; и то и другое — чужое пятно внутри маски.
    """
    image, hair = _scene()

    flat, _ = cheeks.flatten(image, hair, 0.05, 1.0)

    on_face = float(np.mean(flat[_ON_FACE]))
    over_sky = float(np.mean(flat[_GHOST]))

    assert abs(on_face - _SKIN) < abs(on_face - _SKY), "щека залита фоном"
    assert abs(over_sky - _SKY) < abs(over_sky - _SKIN), "фон залит кожей"
    assert over_sky - on_face > 20, "тона не разошлись — заливка вышла одноцветной"


def test_the_two_tones_are_measured_apart():
    """Оба тона уезжают в мету: по ним видно, чем именно залито."""
    image, hair = _scene()

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    skin = float(np.mean(meta["tone_positive"]))
    sky = float(np.mean(meta["background_positive"]))

    assert abs(skin - _SKIN) < 20, "тон кожи замерен мимо кожи"
    assert abs(sky - _SKY) < 20, "тон фона замерен мимо фона"
    assert meta["background_px"][0] > 0 and meta["background_px"][1] > 0


def test_the_hair_class_is_not_overwritten_by_either_tone():
    """
    Исключение, которое двухтоновость не отменяет. Класс волос — это силуэт
    старой копны, и он у редактора единственное указание, где растут волосы:
    стирание, снёсшее его, дало ёжик прямо по черепу. Заливка его не трогает
    ни тоном кожи, ни тоном фона — только размытие.
    """
    image, hair = _scene()
    # Копна на месте пряди над фоном: теперь сегментатор её ВИДИТ
    hair.parsed.hair[_GHOST] = 255

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    coiffure = float(np.mean(flat[_GHOST]))
    assert abs(coiffure - _SKY) > 40, "силуэт копны залит фоном"


def test_the_clothes_keep_their_own_tone():
    """
    Второе исключение: цвет неба на вороте — то же чужое пятно внутри маски,
    что и цвет щеки. Одежда остаётся на размытии.
    """
    image, hair = _scene()
    collar = (slice(250, 275), slice(70, 82))
    image[collar] = 60
    hair.parsed.clothes[collar] = 255

    flat, _ = cheeks.flatten(image, hair, 0.05, 1.0)

    assert abs(float(np.mean(flat[collar])) - _SKY) > 40, "ворот залит небом"


def test_the_jaw_clip_does_not_block_the_background_fill():
    """
    Отсечка по челюсти гасит ТОЛЬКО тон кожи: ниже подбородка лежит шея с
    чужим освещением. Фон сбоку от шеи — это тот же фон и та же прядь, и
    перезаписывать его надо на всей высоте полосы.
    """
    image, hair = _scene()
    below = (slice(250, 275), slice(70, 82))
    image[below] = _STRAND

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["jaw_clipped"] is True
    patch = float(np.mean(flat[below]))
    assert abs(patch - _SKY) < abs(patch - _STRAND), "фон под челюстью остался тёмным"


def test_the_global_background_is_the_fallback():
    """
    Краевой случай из ТЗ: рядом с полосой чистого фона нет — всё занято
    персонажем. Тогда берётся медиана фона по всему кадру, а не отказ: без
    второго тона призрак возвращается.
    """
    image, hair = _scene()
    # Всё объявлено одеждой, кроме лица и угла кадра. Угол выбран ЗА пределами
    # расширенной рамки полосы: иначе локальный сэмпл нашёлся бы там же, и
    # запасной путь остался бы непроверенным
    hair.parsed.clothes[:] = 255
    hair.parsed.clothes[hair.parsed.face > 0] = 0
    hair.parsed.clothes[0:20, 0:8] = 0

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["background_px"] == [0, 0], "локальный сэмпл всё-таки нашёлся"
    assert meta["background_positive"] is not None, "запасной путь не сработал"


def test_nothing_changes_outside_the_mask():
    """
    Обещание, на котором держится безопасность приёма: испорченные пиксели не
    могут попасть в готовый разворот, потому что вклейка берёт всё, что вне
    маски, из нетронутого кадра.
    """
    image, hair = _scene()

    flat, _ = cheeks.flatten(image, hair, 0.05, 1.0)

    outside = hair.mask == 0
    assert np.array_equal(flat[outside], image[outside])


def test_the_flat_ratio_trades_the_dark_patch_for_the_gradient():
    """
    Цена ручки, и она обязана быть видимой. Полный вес заливки поднимает прядь
    до тона кожи; вес поменьше сохраняет низкие частоты щеки, но вместе с ними
    оставляет на месте пряди мягкое тёмное пятно — то есть частично возвращает
    дефект, ради которого всё написано. Оператор выбирает между картонной щекой
    и остаточным пятном, и выбирать он должен, зная обе стороны.
    """
    image, hair = _scene()

    flat, _ = cheeks.flatten(image, hair, 0.05, 1.0)
    soft, _ = cheeks.flatten(image, hair, 0.05, 0.4)

    strand = (slice(150, 220), slice(104, 114))
    assert float(flat[strand].mean()) > float(soft[strand].mean())
    assert float(soft[strand].mean()) > float(image[strand].mean()), "прядь не тронута вовсе"


def test_without_landmarks_the_frame_is_left_alone():
    """
    Разворот без сетки лица строится по разметке, и шести опорных точек не
    существует. Мерить тон нечем — отказ, а не догадка наугад.
    """
    image, hair = _scene()
    hair.points = None

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta == {"flattened": False, "reason": "no_landmarks"}
    assert flat is image


def test_zero_switches_the_whole_thing_off():
    """Ноль возвращает тот же объект: редактору уезжает исходный кадр."""
    image, hair = _scene()

    flat, meta = cheeks.flatten(image, hair, 0.0, 1.0)

    assert meta == {"flattened": False, "reason": "off"}
    assert flat is image


def test_a_band_eaten_by_the_face_guard_is_reported():
    """
    Полоса, до маски не дошедшая, — это не «сработало вхолостую», а причина,
    записанная в мету: крутить надо защиту лица, а не сигму размытия.
    """
    image, hair = _scene()
    hair.mask = np.zeros_like(hair.mask)

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["reason"] == "closed"
    assert flat is image


def test_an_aggressive_band_does_not_starve_the_sample():
    """
    Регрессия, пойманная сквозным прогоном, и молчаливая: шаг выключался сам,
    записав `no_tone`.

    Тон кожи запрещено мерить внутри полосы — иначе прочитаем прядь и зальём её
    же цветом. Пока кромка стояла снаружи от опорных точек щеки, диск попадал в
    чистую кожу медиальнее неё. Стоило опустить кромку на висок (против корня
    пряди у глаза), и диск оказался ВНУТРИ полосы целиком: выборка обнулилась,
    и вместе с ней выключилась вся деструкция.

    Диск обязан сдвигаться к оси, а не сдаваться.
    """
    image, hair = _scene()
    # Полоса до 25 пикселей от оси — заведомо глубже опорных точек щеки (42)
    axis = int(_AXIS)
    hair.cheeks[:] = 0
    hair.cheeks[102:282, axis - 108 : axis - 25] = 255
    hair.cheeks[102:282, axis + 25 : axis + 108] = 255

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["flattened"] is True, f"шаг выключился: {meta.get('reason')}"
    assert min(meta["sample_px"]) >= 24, "сэмпл голодает под агрессивной полосой"
    assert abs(float(np.mean(meta["tone_positive"])) - _SKIN) < 20, "тон уехал мимо кожи"


def test_the_sample_never_slides_onto_the_nose():
    """
    У сдвига есть пол. Полоса, дошедшая до самой оси, увела бы диск на крыло
    носа, и «тон щеки» пришёл бы оттуда — а там свои блики и своя тень.
    """
    image, hair = _scene()
    axis = int(_AXIS)
    hair.cheeks[:] = 0
    hair.cheeks[102:282, axis - 108 : axis + 108] = 255  # полоса накрыла всё

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    # Читать неоткуда — но модуль обязан честно сказать это, а не подсунуть нос
    assert meta["flattened"] is False and meta["reason"] == "no_tone"


# --- Прорыв под челюсть по классу волос ---------------------------------------
#
# Ни на одном шаблоне в репозитории класса волос ниже челюсти нет вовсе: все
# персонажи коротко стрижены (замер — 0 пикселей при 81-168 тысячах волос
# всего). Симптом приходит с профильных кадров с густой причёской, и
# воспроизводится он только сценой, построенной руками.

# Все три области лежат ВНУТРИ левой полосы (столбцы 52-115) — иначе проверки
# проходили бы по той причине, что до них просто не дотянулась маска.
#
# Хвост причёски под ухом: ниже челюсти, снаружи от раздутой кожи шеи.
_TAIL = (slice(250, 285), slice(58, 78))
# Прядь, лежащая НА шее: ниже челюсти, но по коже — обязана уцелеть.
_ON_NECK = (slice(250, 285), slice(104, 114))
# Прядь на виске: выше челюсти, силуэт копны — тоже обязана уцелеть.
_ABOVE_JAW = (slice(150, 200), slice(66, 86))


def _long_hair():
    """Сцена с длинными волосами: хвост под ухом, прядь на шее, прядь на виске."""
    image, hair = _scene()

    # Шея: голая кожа ниже подбородка. Широкая намеренно — она обязана дойти до
    # полосы, иначе защита шеи проверялась бы там, куда маска и так не доходит
    hair.parsed.skin[240:300, 100:220] = 255
    image[240:300, 100:220] = _SKIN

    for region in (_TAIL, _ON_NECK, _ABOVE_JAW):
        image[region] = _STRAND
        hair.parsed.hair[region] = 255
        hair.parsed.face[region] = 0
        hair.parsed.skin[region] = 0
    return image, hair


def test_the_tail_below_the_jaw_is_burned():
    """
    Сам дефект. Хвост причёски под ухом лежит ниже челюсти: силуэта копны там
    уже нет, есть то, что мы срезаем. Прежде он не попадал ни в одну зону и
    получал одно размытие — на профиле оставались тёмные пряди.
    """
    image, hair = _long_hair()

    flat, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    tail = float(np.mean(flat[_TAIL]))
    assert abs(tail - _SKY) < abs(tail - _STRAND), "хвост под ухом остался тёмным"
    assert meta["filled_neck_hair_px"] > 0, "прорыв под челюсть не сработал"


def test_the_silhouette_above_the_jaw_is_still_untouchable():
    """
    Прорыв идёт строго ВНИЗ. Выше челюсти класс волос по-прежнему заливкой не
    трогается: это силуэт копны, и стирание, снёсшее его, дало ёжик по черепу.
    """
    image, hair = _long_hair()

    flat, _ = cheeks.flatten(image, hair, 0.05, 1.0)

    temple = float(np.mean(flat[_ABOVE_JAW]))
    assert abs(temple - _SKY) > 40, "силуэт копны на виске залит фоном"


def test_the_bare_neck_keeps_its_protection():
    """
    Прядь, лежащая на голой коже шеи, из прорыва исключена: тон фона на шее —
    то же чужое пятно внутри маски, от которого ушла двухтоновая заливка.
    """
    image, hair = _long_hair()

    flat, _ = cheeks.flatten(image, hair, 0.05, 1.0)

    on_neck = float(np.mean(flat[_ON_NECK]))
    assert abs(on_neck - _SKY) > 40, "прядь на шее залита фоном"


def test_the_collar_is_excluded_from_the_breakthrough():
    """Одежда исключена так же явно, как и шея: цвет неба на вороте недопустим."""
    image, hair = _long_hair()
    hair.parsed.clothes[_TAIL] = 255
    hair.parsed.hair[_TAIL] = 0

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["filled_neck_hair_px"] == 0, "прорыв пошёл по вороту"


def test_the_breakthrough_needs_the_jaw_line():
    """
    Без линии челюсти «ниже» не определено. Прорыва тогда нет — расширяться
    вниз наугад значит заливать шею фоном.
    """
    image, hair = _long_hair()
    hair.points = list(hair.points)
    hair.points[150] = hair.points[379] = hair.points[152]  # хорда выродилась

    _, meta = cheeks.flatten(image, hair, 0.05, 1.0)

    assert meta["jaw_clipped"] is False
    assert meta["filled_neck_hair_px"] == 0
