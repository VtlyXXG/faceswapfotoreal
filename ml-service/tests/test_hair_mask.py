"""
Маска волос: область первого шага двухшаговой стратегии.

Проверяется одно свойство важнее всех остальных: **лицо остаётся вне маски**.
За ним идёт фейссвоп, и он ищет глаза, нос и рот своим детектором на уже
поправленном шаблоне; тронутое диффузией лицо — это риск молчаливого отказа и
сглаженной кожи, то есть ровно тех двух провалов, из-за которых сюда и пришли.

Разметка в тестах поддельная: настоящая требует весов на 16 МБ и настоящего
человека в кадре, а модулю достаточно четырёх масок классов.
"""

import numpy as np
import pytest

from app.core.errors import InvalidImageError, NoFaceDetectedError
from app.pipelines import hair_mask, head_mask, parsing

_SIZE = 400
_DEFAULT = object()


@pytest.fixture
def image():
    return np.full((_SIZE, _SIZE, 3), 180, dtype=np.uint8)


def _parsed(hair_box, face_box, clothes_box=None, skin_box=None) -> parsing.Parsed:
    """
    Разметка из прямоугольников: (top, bottom, left, right).

    Классы не пересекаются, и это не удобство теста, а свойство настоящей
    разметки: сегментатор отдаёт категорию на пиксель, а не набор слоёв. Пиксель
    чёлки — либо волосы, либо лицо, и решает это модель, а не мы.

    Приоритет здесь у лица: прямоугольник волос описывает всю голову, лицо
    вырезается из него, и получается причёска в виде рамки вокруг лица — именно
    так разметка и разбирает голову. Где нужно обратное (чёлка на лбу), тест
    опускает верхнюю границу лица и оставляет лоб волосам.
    """

    def mask(box) -> np.ndarray:
        plane = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        if box is not None:
            top, bottom, left, right = box
            plane[top:bottom, left:right] = 255
        return plane

    hair, face, skin, clothes = (mask(box) for box in (hair_box, face_box, skin_box, clothes_box))
    hair = np.where(face > 0, 0, hair)
    skin = np.where((hair > 0) | (face > 0), 0, skin)
    clothes = np.where((hair > 0) | (face > 0) | (skin > 0), 0, clothes)

    return parsing.Parsed(face=face, hair=hair, skin=skin, clothes=clothes)


@pytest.fixture
def stub(monkeypatch, mesh):
    """
    Сетка лица и разметка вокруг центра кадра.

    Лицо — 80 пикселей высотой (см. conftest), причёска накрывает его сверху и
    свисает ниже подбородка: ровно тот случай, ради которого всё написано, —
    длинные волосы, которые надо укоротить.
    """

    default = _parsed(
        # Волосы: шапка над лицом плюс длинные пряди до плеч
        hair_box=(110, 320, 130, 270),
        face_box=(160, 260, 160, 240),
        clothes_box=(300, 400, 100, 300),
    )

    def setup(points=None, parsed=_DEFAULT):
        # Сравнение с сентинелом, а не с None: None здесь — законное значение,
        # им и проверяется путь «весов разметки нет»
        landmarks = points if points is not None else mesh(centre=(200, 200))
        monkeypatch.setattr(head_mask, "try_landmarks", lambda _: landmarks)
        monkeypatch.setattr(parsing, "parse", lambda _: default if parsed is _DEFAULT else parsed)
        return landmarks

    return setup


def _build(image, **overrides):
    kwargs = {
        "dilate_ratio": 0.10,
        "feather_ratio": 0.07,
        "protect_ratio": 0.05,
        "forehead_ratio": 0.5,
        "core_ratio": 1.0,
        "guard_ratio": 1.0,
        "cheek_ratio": 0.0,
    }
    kwargs.update(overrides)
    return hair_mask.build(image, **kwargs)


def test_the_face_stays_out_of_the_mask(image, stub):
    """
    Главное свойство этой маски. Следом идёт фейссвоп, и лицо ему нужно
    нетронутым: своим детектором он ищет его на поправленном шаблоне, а не
    найдя — молча вернёт кадр без замены.
    """
    stub()

    mask = _build(image).mask

    # Центр лица, глаза и рот — в координатах заглушки из conftest
    assert mask[200, 200] == 0, "центр лица под маской"
    assert mask[190, 170] == 0 and mask[190, 230] == 0, "глаза под маской"
    assert mask[238, 200] == 0, "рот под маской"


def test_the_hair_itself_is_open(image, stub):
    """Ради чего маска и строится: волосы модели отдаются целиком."""
    stub()

    mask = _build(image).mask

    assert mask[130, 200] == 255, "шапка волос над лицом"
    assert mask[300, 150] == 255, "прядь ниже подбородка"


def test_the_trace_of_the_old_hair_is_included(image, stub):
    """
    Длинные волосы лежат на плечах и закрывают одежду. Короткая стрижка их
    открывает, и то, что было под ними, модель обязана дорисовать — значит,
    след старой причёски входит в область целиком, вместе с задетой тканью.
    Число уезжает в мету: если перерисован весь жилет, объяснит это оно.
    """
    stub()

    result = _build(image)

    assert result.meta["clothes_px"] > 0, "волосы лежат на одежде — ткань под маской"
    # Но не вся ткань: маска идёт по следу волос, а не по костюму
    assert result.meta["clothes_px"] < 100 * 200


def test_the_mask_grows_outward_only(image, stub):
    """
    Расширение и растушёвка идут наружу, в фон: сами волосы обязаны остаться под
    сплошными 255. Полупрозрачные волосы означают призрак старой причёски по
    контуру — он переживает любую вклейку, потому что лежит внутри маски.
    """
    stub()

    wide = _build(image, dilate_ratio=0.4)
    narrow = _build(image, dilate_ratio=0.0)

    assert wide.meta["open_px"] > narrow.meta["open_px"]
    assert wide.mask[130, 200] == 255 and narrow.mask[130, 200] == 255


def test_a_fringe_over_the_brows_is_still_hair(image, stub):
    """
    Защита вычитает контур лица МИНУС волосы. Прядь, упавшая на бровь, — часть
    причёски: оставить её значит оставить кусок старой гривы над глазом.
    """
    stub(
        parsed=_parsed(
            # Лоб отдан волосам: разметка отнесла чёлку к причёске, и класс лица
            # начинается ниже бровей — то есть внутри контура из сетки
            hair_box=(110, 200, 130, 270),
            face_box=(200, 260, 160, 240),
        )
    )

    mask = _build(image).mask

    assert mask[175, 200] == 255, "чёлка на лбу правится вместе с причёской"
    assert mask[230, 200] == 0, "кожа лица под тем же контуром защищена"
    # А вот прядь, доставшая до глаза, остаётся: ядро лица волосам не уступает.
    # Глаза заглушки на y=190 (conftest), и это единственное место, где старая
    # причёска имеет право уцелеть
    assert mask[190, 170] <= 8, "у самого угла глаза ядро гасит маску не в ноль, а почти"


def test_the_sideburns_are_not_rescued_by_the_face_guard(image, stub):
    """
    Правка после четвёртого прогона: на висках и лбу повисли ошмётки старых
    тёмных волос.

    Защита лица расширяется НАРУЖУ — на protect_ratio во все стороны, то есть в
    прилегающие волосы. Спасённую ею прядь редактор не сотрёт, а фейссвоп вклеит
    лицо под неё. Поэтому готовый вес защиты гасится классом волос: пиксель,
    размеченный как волосы, обязан попасть под правку, даже если лежит на щеке.

    Бакенбарда здесь — полоса волос вплотную к щеке, внутри контура лица.
    """
    parsed = _parsed(hair_box=(110, 320, 130, 270), face_box=(160, 260, 160, 240))
    parsed.hair[244:258, 160:178] = 255  # прядь на щеке, внутри класса лица
    parsed.face = np.where(parsed.hair > 0, 0, parsed.face)
    stub(parsed=parsed)

    # Поле защиты шире пряди: без вычитания волос она уцелела бы целиком
    result = _build(image, protect_ratio=0.1)

    assert result.mask[251, 169] == 255, "бакенбарда идёт под правку"
    assert result.mask[251, 200] == 0, "щека рядом с ней — нет"
    assert result.meta["hair_left_px"] < result.meta["hair_px"] * 0.02


def test_narrowing_the_core_frees_the_temple_but_not_the_eye(image, stub):
    """
    Последняя прядь: локон, спускающийся от виска к щеке.

    Оболочка ядра кончается на внешнем углу глаза, но дилатация уводит её ещё на
    `margin` в сторону виска — и локон оказывается под защитой, которую волосы
    не перебивают. `core_ratio` сжимает оболочку поперёк оси лица, освобождая
    висок.

    Безопасность держится не на осторожности значения, а на устройстве:
    дилатация круговая и возвращает накрытие наружу, поэтому сам угол глаза
    остаётся внутри ядра. Проверяется здесь и то, и другое — иначе «сузили» и
    «открыли глаз диффузии» станут одним и тем же коммитом.

    Внешние углы глаз заглушки на x=170 и x=230, ось лица по x=200 (conftest).
    """
    parsed = _parsed(hair_box=(110, 320, 130, 270), face_box=(160, 260, 160, 240))
    parsed.hair[185:215, 160:169] = 255  # прядь на виске, вплотную к глазу
    parsed.face = np.where(parsed.hair > 0, 0, parsed.face)
    stub(parsed=parsed)

    wide = _build(image, protect_ratio=0.1, core_ratio=1.0)
    narrow = _build(image, protect_ratio=0.1, core_ratio=0.7)

    assert wide.mask[200, 165] < 128, "ядро целиком дотягивается до виска"
    # Не 255: хвост размытия ядра сюда всё же дотягивается — но это уже правка,
    # а не защита. Ровно ноль тут и не нужен, нужен вес, при котором вклейка
    # берёт пиксель из генерации
    assert narrow.mask[200, 165] > 200, "сжатое — уже нет, прядь идёт под правку"

    # А глаз под защитой в обоих случаях: сдвиг угла (0.3 × 30 = 9) меньше
    # поля дилатации (0.1 × 82 = 8)… почти, и потому проверяется, а не считается
    assert narrow.mask[190, 180] <= 8, "внутренняя часть глаза защищена"
    assert narrow.mask[210, 200] <= 8, "и лицо между глазом и носом тоже"
    assert narrow.meta["core_ratio"] == 0.7


def test_narrowing_the_guard_frees_a_strand_the_parsing_calls_a_face(image, stub):
    """
    Прядь, которую разметка отнесла к классу ЛИЦА, а не волос.

    Так теряется тонкий тёмный локон от виска к щеке: сегментатор работает на
    256x256 и растворяет его в классе FACE. Дальше он недостижим ничем из
    прежних чисел — и это здесь и проверяется. В маску он не входит: маска
    строится из класса волос. Гашение защиты волосами его не касается по той же
    причине. `core` жмёт ЯДРО, то есть оболочку глаз, носа и рта, а класс FACE
    остаётся во всю щёку. И, главное, расширять за ним маску бесполезно: защита
    вычитается ПОСЛЕ дилатации, и на щеке маска умножится на ноль при любом
    dilate_ratio.

    Остаётся сузить сам источник защиты. Ось лица по x=200, углы глаз на x=170
    и x=230 (conftest) — и глаза обязаны пережить сужение, иначе «освободили
    щёку» и «пустили диффузию по глазам» станут одним коммитом.
    """
    parsed = _parsed(hair_box=(110, 320, 130, 270), face_box=(160, 260, 160, 240))
    # Локон на щеке, размеченный ЛИЦОМ: класс face выходит за свой
    # прямоугольник влево, класс hair на этом месте стёрт
    parsed.face[215:250, 142:153] = 255
    parsed.hair = np.where(parsed.face > 0, 0, parsed.hair)
    stub(parsed=parsed)

    kept = _build(image, protect_ratio=0.05)
    wider = _build(image, protect_ratio=0.05, dilate_ratio=0.4)
    freed = _build(image, protect_ratio=0.05, guard_ratio=0.6)

    assert kept.mask[230, 147] <= 8, "прядь под защитой: маска её не берёт"
    # Ровно то, ради чего тест и написан: расширение маски здесь не работает и
    # работать не может — защита отнимается после него
    assert wider.mask[230, 147] <= 8, "вчетверо более широкая маска пряди не достаёт"
    assert freed.mask[230, 147] == 255, "сужённая защита отдаёт прядь редактору"

    # Глаза и центр лица переживают сужение: ядро возвращается поверх среза
    assert freed.mask[190, 170] <= 8 and freed.mask[190, 230] <= 8, "глаза защищены"
    assert freed.mask[200, 200] == 0 and freed.mask[238, 200] == 0, "лицо и рот тоже"
    assert freed.meta["guard_ratio"] == 0.6
    assert freed.meta["protected_px"] < kept.meta["protected_px"]
    # Полуширина меряется по источнику ДО среза — иначе подбор гонялся бы за
    # собственным хвостом: каждое сужение уменьшало бы и то, от чего берётся доля
    assert freed.meta["guard_half_px"] == kept.meta["guard_half_px"]
    # А граница ответственности двух ручек — сужается вместе с защитой
    assert freed.meta["guard_reach_px"] < kept.meta["guard_reach_px"]


def _lost_strand(stub):
    """
    Разметка, в которой локон на щеке не отнесён НИ К ЧЕМУ.

    Третий, и худший, случай потерянной пряди. Первые два — прядь в классе волос,
    спасённая защитой (её берёт `protect`/`core`), и прядь, отнесённая к классу
    ЛИЦА (её берёт `guard`). Здесь же сегментатор не видит на этих пикселях
    ничего: ни волос, ни лица, ни кожи, ни одежды. Так на 256x256 и пропадает
    тонкий тёмный локон от виска к воротнику.

    Причёска поэтому оставлена шапкой над лицом — без длинных прядей, чтобы класс
    волос до щеки не доставал вовсе, — а сам локон в разметке отсутствует.
    """
    parsed = _parsed(hair_box=(110, 175, 130, 270), face_box=(160, 260, 160, 240))
    stub(parsed=parsed)
    return parsed


def test_a_strand_the_parsing_does_not_see_at_all(image, stub):
    """
    Ровно то, что показал прогон со стиранием: локон вне маски целиком.

    И, главное, второе — почему это не было видно в логах. `hair_left_px` меряет
    остаток КЛАССА волос, а класс этой пряди не знает: ноль в нём прекрасно
    уживается с локоном во всю щёку. Пока это не проверено, следующий такой
    случай снова будут искать в редакторе.
    """
    _lost_strand(stub)

    result = _build(image)

    assert result.mask[270, 150] == 0, "полосы нет — маска до щеки не достаёт"
    # Единицы пикселей — те самые «в пределах нормы», что были в логе живого
    # прогона. Локон во всю щёку при этом висит нетронутым: метрика его не видит
    assert result.meta["hair_left_px"] < 10, "и метрика остатка об этом молчит"


def test_the_cheek_band_swallows_it(image, stub):
    """
    Единственное, чем такая прядь берётся: область, заданная геометрией.

    Ничто из остальных чисел до неё не дотягивается по построению — и это
    проверяется здесь же, вместе с самой полосой: расширять нечего (источника в
    разметке нет), а `guard` и `core` только отпускают защиту, области не
    добавляя.
    """
    _lost_strand(stub)

    wider = _build(image, dilate_ratio=0.4)
    freed = _build(image, guard_ratio=0.3, core_ratio=0.5)
    band = _build(image, cheek_ratio=0.35)

    assert wider.mask[270, 150] == 0, "вчетверо более широкая маска щеки не достаёт"
    assert freed.mask[270, 150] == 0, "отпущенная защита области не добавляет"
    assert band.mask[270, 150] == 255, "полоса вдоль щеки — достаёт"


def test_the_cheek_band_outlives_the_face_guard(image, stub):
    """
    Полоса, не гасящая защиту, не работала бы вовсе: класс FACE растянут на всю
    щёку, а защита вычитается ПОСЛЕ дилатации — на щеке маска умножилась бы на
    ноль. Поэтому полоса гасит защиту наравне с настоящими волосами.

    Точка взята вплотную к классу лица (он начинается с x=160), то есть там, где
    поле защиты заведомо накрывает.
    """
    _lost_strand(stub)

    result = _build(image, cheek_ratio=0.35, protect_ratio=0.05)

    # Не ровно 255: точка стоит у самой внутренней кромки полосы, и туда
    # дотягивается хвост размытия защиты. Нужен не ноль в защите, а вес, при
    # котором вклейка берёт пиксель из генерации, — он здесь и есть
    assert result.mask[230, 158] > 200, "полоса переживает защиту вплотную к лицу"
    assert result.mask[230, 140] == 255, "а глубже в полосе защиты нет вовсе"


def test_the_cheek_band_never_opens_the_face(image, stub):
    """
    Цена полосы — щёки и шея под диффузией; лицо в эту цену не входит ни при
    каком её размере. Держат его два независимых обстоятельства, и проверяются
    оба: ядро лица накладывается поверх всего, а сама полоса начинается снаружи
    от внешних углов глаз и внутрь не заходит.
    """
    _lost_strand(stub)

    huge = _build(image, cheek_ratio=2.0)

    assert huge.mask[190, 170] <= 8 and huge.mask[190, 230] <= 8, "глаза защищены"
    assert huge.mask[200, 200] == 0, "центр лица"
    assert huge.mask[238, 200] == 0, "рот"


def test_the_cheek_band_turns_with_the_head(image, stub, mesh):
    """
    Полоса живёт в осях головы, а не кадра, и по той же причине, что срез лба и
    сужение защиты: у персонажа, склонившего голову набок, вертикальные полосы
    прошли бы по щеке с одной стороны и по фону с другой.

    Точка берётся не на глаз, а считается по той же геометрии: на щеке, ниже
    подбородка и в стороне от оси.
    """
    points = mesh(centre=(200, 200), angle=25.0)
    cap = _parsed(hair_box=(110, 175, 130, 270), face_box=(160, 260, 160, 240))
    stub(points=points, parsed=cap)

    geometry = head_mask.face_geometry(points)
    chin, up, side = geometry["chin"], geometry["up"], geometry["side"]
    spot = chin - up * (geometry["face_height"] * 0.1) + side * (geometry["face_width"] * 0.45)
    column, row = int(round(spot[0])), int(round(spot[1]))

    result = _build(image, cheek_ratio=0.35)

    assert result.mask[row, column] == 255, "полоса повернулась вместе с головой"


def test_the_cheek_band_reports_what_the_parsing_thought(image, stub):
    """
    Диагностика, которой не хватило и которая стоила прогона. Большая площадь
    полосы при почти нулевом `cheek_hair_px` — это и есть «разметка волос тут не
    видит», то есть подтверждение, что полоса здесь единственный способ. А
    `cheek_open_px` заметно меньше `cheek_px` означал бы, что полосу съела
    защита, и вот тогда крутить надо `guard`.
    """
    _lost_strand(stub)

    off = _build(image).meta
    on = _build(image, cheek_ratio=0.35).meta

    assert off["cheek_px"] is None and off["cheek_open_px"] is None
    assert on["cheek_ratio"] == 0.35
    assert on["cheek_px"] > 0
    assert on["cheek_hair_px"] < on["cheek_px"] * 0.05, "разметка волос тут не видит"
    assert on["cheek_open_px"] > on["cheek_px"] * 0.5, "и защита полосу не съела"


def test_the_leftover_hair_says_which_knob_holds_it(image, stub):
    """
    Остаток волос без места неинформативен: держат его либо суженная защита,
    либо ядро, а крутить их надо в разные стороны. Ядро накладывается ПОСЛЕ
    среза по ширине, поэтому дотягивается дальше него — по выносу остатка и
    видно, кто виноват.

    Прядь здесь лежит поверх глаза: это ровно то место, где ядро не уступает
    волосам никогда, и единственный остаток, который считается нормой.
    """
    parsed = _parsed(hair_box=(110, 320, 130, 270), face_box=(160, 260, 160, 240))
    parsed.hair[185:196, 165:176] = 255  # прядь на внешнем углу глаза
    parsed.face = np.where(parsed.hair > 0, 0, parsed.face)
    stub(parsed=parsed)

    result = _build(image, protect_ratio=0.05, guard_ratio=0.6)

    near, far = result.meta["hair_left_span_px"]
    assert result.meta["hair_left_px"] > 0, "прядь поверх глаза ядро не отдаёт"
    assert far > near >= 0
    # Держит её ядро, а не защита: она лежит дальше, чем достаёт срезанная защита
    assert far > result.meta["guard_reach_px"]


def test_no_leftover_hair_means_no_place_to_report(image, stub):
    """Пустой остаток — это None, а не [0, 0]: нуль здесь читался бы как «у оси»."""
    stub()

    result = _build(image, guard_ratio=0.6)

    assert result.meta["hair_left_px"] == 0
    assert result.meta["hair_left_span_px"] is None


def test_the_forehead_can_be_handed_over_to_the_editor(image, stub):
    """
    Правка после третьего прогона: над бровями шёл рубец.

    Пока лоб защищён целиком, нижняя граница маски проходит по открытой коже, а
    вклейка геометрию не выравнивает — сдвинувшаяся генерация превращает эту
    границу в ступеньку, и голова читается как накладка. Опущенная до бровей
    маска отдаёт лоб редактору целиком: полосы старой кожи между лицом от
    фейссвопа и новой причёской не остаётся.

    Брови у заглушки на y=178, подбородок на y=260 (см. conftest).
    """
    stub()

    seam = _build(image, forehead_ratio=0.0).mask
    kept = _build(image, forehead_ratio=0.5).mask

    assert seam[165, 200] == 255, "лоб отдан редактору"
    assert kept[165, 200] == 0, "при умолчании лоб по-прежнему защищён"
    # Ниже линии бровей защита работает одинаково: диффузии там делать нечего
    assert seam[200, 200] == 0 and seam[238, 200] == 0


def test_the_forehead_hand_over_is_counted(image, stub):
    """
    Ноль в мете при малом forehead_ratio означает, что линия бровей посчитана
    мимо, и рубец останется на месте. Разбираться в этом по картинке дороже.
    """
    stub()

    assert _build(image, forehead_ratio=0.0).meta["forehead_px"] > 0
    assert _build(image, forehead_ratio=0.5).meta["forehead_px"] == 0


def test_the_neighbour_on_the_spread_keeps_his_hair(image, stub):
    """
    Классы разметки приходят одним слоем на всех героев сразу. Перекрашивать
    причёску второго персонажа мы не нанимались — отбор идёт по связности с
    нашим лицом.
    """
    hair = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    hair[110:320, 130:270] = 255  # наш персонаж
    hair[110:200, 300:380] = 255  # сосед по развороту
    parsed = _parsed(hair_box=None, face_box=(160, 260, 160, 240))
    parsed.hair = hair

    stub(parsed=parsed)

    mask = _build(image).mask

    assert mask[130, 200] == 255, "своя причёска открыта"
    assert mask[150, 340] == 0, "чужая — нет"


def test_without_parsing_the_ring_takes_over(image, stub, caplog):
    """
    Без весов разметки формы причёски мы не знаем. Кольцо «эллипс головы минус
    лицо» — затычка, чтобы сервис не отказывал заказу целиком, и в логе она
    обязана быть видна: длинные волосы такое кольцо не накроет.
    """
    stub(parsed=None)

    result = _build(image)

    assert result.meta["source"] == "ellipse"
    assert result.mask[200, 200] == 0, "лицо защищено и на запасном пути"
    assert any("кольцом" in record.message for record in caplog.records)


def test_no_character_at_all_is_refused(image, monkeypatch):
    """
    Ни сетки, ни разметки — маску строить не из чего. 422 с подсказкой: перенос
    одного лица работает и без локальной геометрии.
    """
    monkeypatch.setattr(head_mask, "try_landmarks", lambda _: None)
    monkeypatch.setattr(parsing, "parse", lambda _: None)

    with pytest.raises(NoFaceDetectedError) as exc_info:
        _build(image)

    assert exc_info.value.status_code == 422
    assert "face_swap" in exc_info.value.details["hint"]


def test_protection_eating_the_whole_mask_is_refused(image, stub):
    """
    Защита с огромным полем гасит область целиком. Пустая маска означала бы
    вклейку без единого изменённого пикселя — то есть заказ, прошедший два
    вызова fal впустую.
    """
    stub()

    with pytest.raises(NoFaceDetectedError):
        _build(image, protect_ratio=10.0)


def test_negative_ratios_are_refused(image, stub):
    stub()

    with pytest.raises(InvalidImageError):
        _build(image, dilate_ratio=-0.1)


def test_meta_explains_what_was_built(image, stub):
    stub()

    meta = _build(image).meta

    assert meta["source"] == "parsing"
    assert meta["landmarks"] is True and meta["parsing"] is True
    assert meta["hair_px"] > 0
    assert meta["protected_px"] > 0, "защита лица обязана что-то вычесть"
    assert 0 < meta["open_share"] < 0.35


# --- Полоса на повёрнутой голове ---------------------------------------------
#
# Симметричная полоса живёт ровно до первого разворота: `face_width` сокращается
# проекцией, и полуширина, посчитанная из него, сужает ОБЕ полосы разом — тогда
# как ближняя щека в проекции РАСТЯГИВАЕТСЯ. Ближнюю недокрываем, дальнюю уводим
# на нос. Заглушка сетки умеет поворот (см. conftest), и проверяется он здесь.


def _bands(image, stub, mesh, **turn):
    """Сырая полоса и мета для головы, повёрнутой на заданный угол."""
    stub(points=mesh(centre=(200, 200), **turn))
    built = _build(image, cheek_ratio=0.35)
    return built.cheeks, built.meta


def _side_pixels(bands, geometry, positive: bool) -> int:
    """Сколько пикселей полосы лежит по одну сторону от оси лица."""
    rows, columns = np.nonzero(bands)
    chin, side = geometry["chin"], geometry["side"]
    lateral = (columns - chin[0]) * side[0] + (rows - chin[1]) * side[1]
    return int(np.count_nonzero(lateral >= 0 if positive else lateral < 0))


def test_the_frontal_bands_stay_symmetric(image, stub, mesh):
    """
    Ничего из написанного ниже не должно менять фронтальный кадр: на нём
    прежняя доля полуширины выигрывает у прижима к глазу, и полосы остаются
    двумя зеркальными прямоугольниками. Это защита от регресса — фас работает.
    """
    bands, meta = _bands(image, stub, mesh)
    geometry = head_mask.face_geometry(mesh(centre=(200, 200)))

    assert [band["bound_by"] for band in meta["cheek_bands"]] == ["geometry", "geometry"]
    assert abs(meta["cheek_pose"]["yaw"]) < 0.05

    near, far = meta["cheek_bands"]
    assert near["width_px"] == far["width_px"], "полосы разной ширины на фасе"

    left = _side_pixels(bands, geometry, positive=False)
    right = _side_pixels(bands, geometry, positive=True)
    assert abs(left - right) < max(left, right) * 0.05, "полосы разъехались на фасе"


@pytest.mark.parametrize("degrees", [30, 45, 55, 70])
def test_the_far_cheek_band_survives_the_profile(image, stub, mesh, degrees):
    """
    Регрессия, стоившая композита, и главное свойство этой правки.

    Прежняя версия снимала дальнюю полосу на повороте: «камера не видит эту
    щеку, значит, и правки там не нужно». Рассуждение неверно в самой посылке.
    Маска работает в ПЛОСКОСТИ КАДРА, а прядь на дальней стороне от поворота
    головы никуда не девается — она висит на своём месте. Оставшись вне маски,
    она доезжала до готового разворота нетронутой.

    Поэтому дальняя полоса обязана быть на месте на любом угле, и не тоньше
    доли от фронтальной ширины.
    """
    points = mesh(centre=(200, 200), yaw=degrees)
    bands, meta = _bands(image, stub, mesh, yaw=degrees)
    geometry = head_mask.face_geometry(points)

    assert bands is not None, "полоса исчезла целиком"

    far = next(band for band in meta["cheek_bands"] if band["side"] == "far")
    floor = (
        geometry["face_height"]
        * hair_mask._HALF_PER_HEIGHT
        * (hair_mask._CHEEK_OUTER - hair_mask._CHEEK_INNER)
        * hair_mask._CHEEK_MIN_FAR_SHARE
    )
    assert far["width_px"] >= floor - 0.5, "дальняя полоса уже хард-лимита"

    # И она действительно нарисована, а не только посчитана
    pose = head_mask.head_pose(points, geometry)
    assert (
        _side_pixels(bands, geometry, positive=pose["far"] > 0) > 0
    ), "дальняя полоса посчитана, но в маску не попала"


def test_the_far_band_is_pulled_out_to_the_hair_the_parser_sees(image, stub, mesh):
    """
    Привязка к силуэту. Сетка на повороте показывает на дальней стороне воздух,
    а сегментатор видит там волосы — права разметка, волосы в кадре есть.

    Проверяется сравнением двух разметок на одном и том же повороте: у второй
    копна на дальней стороне длиннее, и кромка полосы обязана уйти за ней.
    """
    turned = mesh(centre=(200, 200), yaw=40)
    geometry = head_mask.face_geometry(turned)
    pose = head_mask.head_pose(turned, geometry)

    # Куда именно вытягивать — зависит от того, какая сторона дальняя
    far_left = pose["far"] < 0
    short = _parsed(hair_box=(110, 320, 130, 270), face_box=(160, 260, 160, 240))
    long = _parsed(
        hair_box=(110, 320, 60, 270) if far_left else (110, 320, 130, 340),
        face_box=(160, 260, 160, 240),
    )

    stub(points=turned, parsed=short)
    tight = _build(image, cheek_ratio=0.35).meta
    stub(points=turned, parsed=long)
    wide = _build(image, cheek_ratio=0.35).meta

    far_tight = next(band for band in tight["cheek_bands"] if band["side"] == "far")
    far_wide = next(band for band in wide["cheek_bands"] if band["side"] == "far")

    assert far_wide["outer_px"] > far_tight["outer_px"], "полоса не пошла за разметкой"
    assert far_wide["bound_by"] == "hair"


def test_the_silhouette_never_pulls_the_band_past_the_frontal_envelope(image, stub, mesh):
    """
    Обратная сторона той же привязки: разметка тянет кромку наружу, но не
    дальше, чем полоса дотягивалась бы в фас. Без этого предела длинная копна
    уводила бы полосу на весь кадр, а размытие — на силуэт причёски, который
    редактору служит единственным указанием, где растут волосы.
    """
    stub(
        parsed=_parsed(
            # Копна во весь кадр — заведомо шире любого фронтального конверта
            hair_box=(20, 380, 10, 390),
            face_box=(160, 260, 160, 240),
        )
    )

    meta = _build(image, cheek_ratio=0.35).meta
    points = mesh(centre=(200, 200))
    geometry = head_mask.face_geometry(points)
    pose = head_mask.head_pose(points, geometry)
    envelope = geometry["face_height"] * hair_mask._HALF_PER_HEIGHT * hair_mask._CHEEK_OUTER

    halves = (pose["near_half"], pose["far_half"])
    for band, half in zip(meta["cheek_bands"], halves, strict=True):
        # Своя геометрия полосу не ограничивает — она её задаёт; предел ставится
        # именно РАЗМЕТКЕ, и потому сравнивается с большим из двух
        assert (
            band["outer_px"] <= max(half * hair_mask._CHEEK_OUTER, envelope) + 0.5
        ), "разметка утащила полосу за конверт"
        assert band["bound_by"] == "geometry", "копна во весь кадр не должна ничего двигать"


def test_the_near_cheek_band_widens_with_the_turn(image, stub, mesh):
    """
    Обратная сторона того же: ближняя щека разворачивается к камере и занимает
    БОЛЬШЕ пикселей, чем в фас. Симметричная полоса, считавшая полуширину из
    сократившегося `face_width`, сужала бы её вместе с дальней.
    """
    _, frontal = _bands(image, stub, mesh)
    _, turned = _bands(image, stub, mesh, yaw=35)

    assert turned["cheek_pose"]["near_half_px"] > frontal["cheek_pose"]["near_half_px"] * 1.1
    assert turned["cheek_pose"]["far_half_px"] < frontal["cheek_pose"]["far_half_px"] * 0.5


@pytest.mark.parametrize("degrees", [0, 10, 20, 30, 40, 50])
def test_the_band_never_climbs_into_the_eye(image, stub, mesh, degrees):
    """
    Ловушка, в которую пропорциональное сужение попадает само. На дальней
    стороне полуширина сжимается быстрее, чем глаз, и доля от неё заезжает под
    веко: внешний угол глаза, замеренный на живой сетке, отходит с 0.71
    полуширины в фас до 2.27 на 35°. Кромка прижата к самому углу глаза, и
    проверяется это на всём диапазоне, а не на одном удобном угле.

    Проверяется именно УГОЛ глаза, и этого достаточно: глаз лежит от него
    внутрь, к оси лица, а полоса — наружу. Чист угол — чист и весь глаз.
    """
    points = mesh(centre=(200, 200), yaw=degrees)
    bands, _ = _bands(image, stub, mesh, yaw=degrees)
    if bands is None:
        return

    for corner in (33, 263):
        column, row = points[corner]
        assert bands[row, column] == 0, f"полоса накрыла внешний угол глаза {corner} на {degrees}°"


def test_without_the_mesh_the_bands_stay_symmetric(image, stub, mesh, monkeypatch):
    """
    Путь через разметку: сетки лица нет, точек нет, мерить поворот нечем.
    Гадать хуже, чем не поправлять, — полоса остаётся прежней, симметричной, и
    мета честно говорит, что позы не было.
    """
    stub()
    with_mesh = _build(image, cheek_ratio=0.35).meta
    assert with_mesh["cheek_pose"] is not None

    monkeypatch.setattr(head_mask, "try_landmarks", lambda _: None)
    without = _build(image, cheek_ratio=0.35).meta

    assert without["cheek_pose"] is None
    assert without["cheek_px"] > 0, "без сетки полоса обязана остаться"
    near, far = without["cheek_bands"]
    assert near["width_px"] == far["width_px"], "без позы полосы обязаны быть зеркальны"


# --- Глаза: полигон вместо кромки --------------------------------------------
#
# Прежде глаз оберегала сама внутренняя кромка полосы: она останавливалась
# снаружи от внешнего угла. Защита была грубой в обе стороны — глаз всё равно
# накрывался на повороте, а корень пряди между углом глаза и кромкой не
# накрывался никогда, и на висках оставался тёмный остаток. Теперь кромка
# опущена на висок, а глаза вычитаются точным полигоном с буфером.


def _eye_extent(points, geometry, ring) -> float:
    """Докуда контур глаза дотягивается вбок от оси, доли полуширины лица."""
    chin, side = geometry["chin"], geometry["side"]
    half = geometry["face_width"] / 2.0
    return max(
        abs(float(np.dot(np.asarray(points[index], dtype=np.float64) - chin, side))) / half
        for index in ring
    )


def test_the_band_now_reaches_the_temple(image, stub, mesh):
    """
    Шаг первый: кромка опущена. Полоса обязана заходить ЗАМЕТНО глубже прежних
    0.75 полуширины — иначе корню пряди по-прежнему негде оказаться внутри неё.
    """
    stub()

    meta = _build(image, cheek_ratio=0.35).meta

    assert meta["cheek_inner_ratio"] == hair_mask._CHEEK_INNER_TEMPLE
    assert meta["cheek_inner_ratio"] < hair_mask._CHEEK_INNER

    points = mesh(centre=(200, 200))
    geometry = head_mask.face_geometry(points)
    for band in meta["cheek_bands"]:
        assert band["inner_px"] < hair_mask._CHEEK_INNER * geometry["face_width"] / 2.0


def test_the_root_between_the_eye_and_the_old_edge_is_covered(image, stub, mesh):
    """
    Сам дефект. Между внешним концом глаза (0.67-0.71 полуширины на живой
    сетке) и прежней кромкой (0.75) оставалась полоска в считанные пиксели — в
    ней и сидел корень пряди на виске, не попадая под заливку ни разу.

    Проверяется, что эта полоска теперь ВНУТРИ полосы: точка сразу снаружи от
    буфера глаза, на уровне глаза, обязана быть накрыта.
    """
    points = stub()
    bands = _build(image, cheek_ratio=0.35).cheeks
    geometry = head_mask.face_geometry(points)

    reach = _eye_extent(points, geometry, hair_mask._EYE_RING_A)
    half = geometry["face_width"] / 2.0
    buffer = geometry["face_height"] * hair_mask._CHEEK_EYE_BUFFER

    # Висок: сразу за буфером глаза, на его же высоте
    chin, side, up = geometry["chin"], geometry["side"], geometry["up"]
    eye_row = points[hair_mask._EYE_RING_A[0]][1]
    along = float(np.dot(np.array([0.0, eye_row]) - chin, up))
    root = chin + up * along - side * (reach * half + buffer + 3)

    assert (
        bands[int(round(root[1])), int(round(root[0]))] == 255
    ), "корень пряди у глаза снова вне полосы"


@pytest.mark.parametrize("degrees", [0, 20, 35])
def test_the_eye_is_carved_out_with_a_buffer(image, stub, mesh, degrees):
    """
    Шаг третий: вычитание не пиксель-в-пиксель. У полосы растушёванная альфа, и
    у самой кромки дыры вес ещё не ноль — размытие затянуло бы туда тьму ресниц
    и зрачка. Поэтому проверяется не только контур, но и кольцо вокруг него.
    """
    points = stub(points=mesh(centre=(200, 200), yaw=degrees))
    built = _build(image, cheek_ratio=0.35)
    geometry = head_mask.face_geometry(points)
    buffer = int(round(geometry["face_height"] * hair_mask._CHEEK_EYE_BUFFER))

    for ring in (hair_mask._EYE_RING_A, hair_mask._EYE_RING_B):
        for index in ring:
            column, row = points[index]
            window = built.cheeks[
                max(0, row - buffer // 2) : row + buffer // 2 + 1,
                max(0, column - buffer // 2) : column + buffer // 2 + 1,
            ]
            assert not window.any(), f"полоса накрыла глаз у точки {index} на {degrees}°"


def test_the_eye_hole_is_reported(image, stub):
    """
    Число в мете. Ноль при построенной сетке означал бы, что кромка до глаз не
    дошла, — то есть корень пряди снова вне заливки, и дефект вернулся молча.
    """
    stub()

    meta = _build(image, cheek_ratio=0.35).meta

    assert meta["cheek_eye_px"] > 0, "полоса до глаз не дошла — вычитать нечего"
    assert meta["cheek_eye_px"] < meta["cheek_px"] * 0.25, "глазам отдана четверть полосы"


def test_without_the_mesh_the_edge_stays_conservative(image, stub, monkeypatch):
    """
    Агрессия кромки лицензирована возможностью вырезать глаза. Полигон строится
    только по сетке; нет сетки — нет и права заходить на висок, иначе полоса
    легла бы на веко без всякой защиты.
    """
    stub()
    with_mesh = _build(image, cheek_ratio=0.35).meta

    monkeypatch.setattr(head_mask, "try_landmarks", lambda _: None)
    without = _build(image, cheek_ratio=0.35).meta

    assert with_mesh["cheek_inner_ratio"] == hair_mask._CHEEK_INNER_TEMPLE
    assert without["cheek_inner_ratio"] == hair_mask._CHEEK_INNER
    assert without["cheek_eye_px"] == 0
