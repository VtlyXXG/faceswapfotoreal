"""
Каталог шаблонов на диске сервера.

Проверяется не «функция вернула список», а три вещи, которыми этот каталог
ломается на практике: порядок страниц (книга читается подряд), отказ на
отсутствующем файле (иначе одиннадцать страниц отвалятся молча) и то, что
идентификатором нельзя прочитать файл вне каталога.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import server


@pytest.fixture()
def catalogue(tmp_path, monkeypatch):
    """
    Книга из трёх страниц на диске: обложка и два разворота.

    Картинки настоящие, а не пустые файлы: `read_template` обязана вернуть
    матрицу, и подделка байтами проверила бы не то.
    """
    book = tmp_path / "dino_pixar_real"
    book.mkdir()
    for name, side in (("cover", 8), ("spread_01", 4), ("spread_02", 4)):
        image = np.full((side, side, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(book / f"{name}.png"), image)

    monkeypatch.setattr(server, "TEMPLATES_ROOT", tmp_path)
    return tmp_path


def test_pages_are_listed_in_reading_order(catalogue):
    """
    Порядок держится на именах, а не на манифесте: обложка впереди разворотов
    просто потому, что `cover` < `spread_01` при обычной сортировке. Если это
    когда-нибудь перестанет быть правдой, книга поедет с середины.
    """
    books = server.template_catalog()

    assert [book["id"] for book in books] == ["dino_pixar_real"]
    assert [page["id"] for page in books[0]["pages"]] == [
        "dino_pixar_real/cover",
        "dino_pixar_real/spread_01",
        "dino_pixar_real/spread_02",
    ]


def test_the_size_of_each_page_is_reported(catalogue):
    """Недокачанный шаблон — самая вероятная беда заливки, и он виден числом."""
    pages = server.template_catalog()[0]["pages"]

    assert all(page["bytes"] > 0 for page in pages)


def test_an_empty_root_is_not_an_error(catalogue, monkeypatch, tmp_path):
    """
    Пересозданная ВМ приходит без шаблонов вовсе. Это состояние, которое надо
    показать в `/health`, а не исключение на опросе здоровья.
    """
    monkeypatch.setattr(server, "TEMPLATES_ROOT", tmp_path / "нет-такого")

    assert server.template_catalog() == []


def test_a_page_that_cannot_be_ordered_is_not_listed(catalogue):
    """
    В поставке имена кириллические, и залитые как есть они дали бы страницу,
    которая в списке видна, а заказать её нельзя: грамматика идентификатора
    такое имя не пропускает. Список обязан показывать только заказуемое.
    """
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(catalogue / "dino_pixar_real" / "разворот_05.png"), image)

    pages = server.template_catalog()[0]["pages"]

    assert all("разворот" not in page["id"] for page in pages)


def test_a_page_is_read_from_disk(catalogue):
    image = server.read_template("dino_pixar_real/cover")

    assert image.shape == (8, 8, 3)


def test_a_missing_page_is_refused_with_404(catalogue):
    """
    404, а не 500: файла нет — это состояние сервера, про которое клиенту есть
    что сказать («такой страницы на сервере нет»), а не сбой рендера.
    """
    with pytest.raises(HTTPException) as exc:
        server.read_template("dino_pixar_real/spread_09")

    assert exc.value.status_code == 404


def test_a_page_is_found_by_any_known_suffix(catalogue):
    """PNG в поставке, но положенный рядом jpg обязан находиться, а не молчать."""
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(catalogue / "dino_pixar_real" / "spread_03.jpg"), image)

    assert server.read_template("dino_pixar_real/spread_03").shape == (4, 4, 3)


def test_a_broken_file_is_refused_loudly(catalogue):
    """
    Мусор вместо картинки — это 500 с именем файла. Молчаливый None от imdecode
    ушёл бы дальше в геометрию и лёг там непонятным отказом.
    """
    (catalogue / "dino_pixar_real" / "spread_04.png").write_bytes(b"not a picture")

    with pytest.raises(HTTPException) as exc:
        server.read_template("dino_pixar_real/spread_04")

    assert exc.value.status_code == 500


def test_a_file_outside_the_root_is_unreachable(catalogue, tmp_path):
    """
    Последний рубеж на случай, если грамматику идентификатора однажды ослабят:
    сам резолвер тоже обязан отказать, а не прочитать соседний файл.
    """
    (tmp_path.parent / "секрет.png").write_bytes(b"x")

    with pytest.raises(HTTPException) as exc:
        server.template_file("../секрет")

    assert exc.value.status_code == 422
