"""Структуры результатов стенда и агрегаты — без зависимостей от моделей."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Variant:
    """Один столбец сравнения."""

    name: str
    enhance: bool
    strength: float = 1.0
    provider: str = "classical"

    @property
    def label(self) -> str:
        if not self.enhance:
            return "без обработки"
        prefix = "" if self.provider == "classical" else f"{self.provider} "
        return f"{prefix}strength {self.strength:g}"


@dataclass
class Cell:
    """Результат одного варианта на одной паре."""

    variant: str
    identity: float | None
    texture: float | None
    crop_uri: str = ""
    note: str = ""


@dataclass
class PairResult:
    pair_id: str
    source_uri: str
    cells: list[Cell] = field(default_factory=list)

    def cell(self, variant: str) -> Cell | None:
        return next((c for c in self.cells if c.variant == variant), None)


@dataclass
class VariantSummary:
    name: str
    label: str
    mean_identity: float | None
    mean_texture: float | None
    below_threshold: int
    n: int


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(
    results: list[PairResult], variants: list[Variant], threshold: float
) -> list[VariantSummary]:
    summaries: list[VariantSummary] = []
    for variant in variants:
        cells = [r.cell(variant.name) for r in results]
        identities = [c.identity for c in cells if c and c.identity is not None]
        textures = [c.texture for c in cells if c and c.texture is not None]
        below = sum(1 for v in identities if v < threshold)
        summaries.append(
            VariantSummary(
                name=variant.name,
                label=variant.label,
                mean_identity=_mean(identities),
                mean_texture=_mean(textures),
                below_threshold=below,
                n=len(identities),
            )
        )
    return summaries


def pick_recommended(
    variants: list[Variant],
    summaries: list[VariantSummary],
    threshold: float,
) -> tuple[Variant | None, str]:
    """
    Рекомендует силу с минимальным рассогласованием фактуры при сохранении
    узнаваемости.

    Не «максимальную силу»: у классического стилизатора фактура немонотонна по
    strength — донорское зерно на больших значениях перебирается, и метрика
    снова растёт. Цель — лучшее согласование, а не самый сильный эффект.
    Отбираем варианты с узнаваемостью не ниже порога и фактурой не хуже, чем
    без обработки, и берём минимум по фактуре; при равенстве — больше стиля.
    """
    by_name = {s.name: s for s in summaries}

    plain = next((v for v in variants if not v.enhance), None)
    plain_texture = by_name[plain.name].mean_texture if plain else None

    eligible = []
    for variant in (v for v in variants if v.enhance):
        summary = by_name[variant.name]
        if summary.mean_identity is None or summary.mean_texture is None:
            continue
        if summary.mean_identity < threshold:
            continue
        if plain_texture is not None and summary.mean_texture > plain_texture:
            continue
        eligible.append(variant)

    if not eligible:
        return None, "ни один вариант не удержал узнаваемость и согласование одновременно"

    # Минимум рассогласования; при равенстве — больше стиля
    best = min(eligible, key=lambda v: (by_name[v.name].mean_texture, -v.strength))
    summary = by_name[best.name]
    reason = (
        f"минимум рассогласования {summary.mean_texture} при узнаваемости "
        f"{summary.mean_identity:.3f} ≥ {threshold:.2f}"
    )
    return best, reason
