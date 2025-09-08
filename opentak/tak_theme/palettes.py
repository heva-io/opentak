from dataclasses import dataclass, field, fields
from typing import Any, TypeAlias

from opentak.tak_theme._colors import (
    BLUE,
    BLUE_CRAYOLA,
    FUSHIA,
    GHOST_WHITE,
    MARGARITA,
    RED_CRAYOLA,
    SAFFRON,
    SUNGLOW,
    TURQUOISE,
)
from opentak.tak_theme._utils import interpolate_colors

__all__ = ("binary", "diverging", "qualitative", "sequential")


class Spreadable(list):
    def spread(self, item: int):
        colors, steps = interpolate_colors(
            color_list=[it[1] for it in self], nb_colors=item, return_steps=True
        )
        return list(zip(steps, colors, strict=False))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Spreadable(super().__getitem__(key))
        return super().__getitem__(key)


# Update 01/10/2024 - Change of TypeAlias required by mypy (in other packages)
Palette: TypeAlias = Spreadable  # Palette: TypeAlias = list[tuple[float, str]]


@dataclass
class ReversableMixin:
    def __getattr__(self, name: str) -> Any:
        if name.endswith("_r") and name.removesuffix("_r") in [
            f.name for f in fields(self)
        ]:
            return super().__getattribute__(name.removesuffix("_r"))[::-1]
        return super().__getattribute__(name)


@dataclass
class _Binary(ReversableMixin):
    onoff: list[str] = field(default_factory=lambda: [MARGARITA, RED_CRAYOLA])
    highlight_blue: list[str] = field(default_factory=lambda: [BLUE_CRAYOLA, BLUE])
    neutral: list[str] = field(default_factory=lambda: [BLUE_CRAYOLA, MARGARITA])


@dataclass
class _Qualitative(ReversableMixin):
    default: list[str] = field(
        default_factory=lambda: [
            BLUE_CRAYOLA,
            RED_CRAYOLA,
            MARGARITA,
            SAFFRON,
            FUSHIA,
            SUNGLOW,
            BLUE,
            TURQUOISE,
        ]
    )


@dataclass
class _Sequential(ReversableMixin):
    blues: Palette = field(
        default_factory=lambda: Spreadable([(0.0, GHOST_WHITE), (1.0, BLUE)])
    )
    light_blues: Palette = field(
        default_factory=lambda: Spreadable([(0.0, GHOST_WHITE), (1.0, BLUE_CRAYOLA)])
    )
    turquoises: Palette = field(
        default_factory=lambda: Spreadable([(0.0, GHOST_WHITE), (1.0, TURQUOISE)])
    )
    greens: Palette = field(
        default_factory=lambda: Spreadable([(0.0, GHOST_WHITE), (1.0, MARGARITA)])
    )
    reds: Palette = field(
        default_factory=lambda: Spreadable([(0.0, GHOST_WHITE), (1.0, RED_CRAYOLA)])
    )
    blue_turquoise: Palette = field(
        default_factory=lambda: Spreadable(
            [(0.0, BLUE), (0.5, BLUE_CRAYOLA), (1.0, TURQUOISE)]
        )
    )
    blue_saffron: Palette = field(
        default_factory=lambda: Spreadable([(0.0, BLUE), (0.5, FUSHIA), (1.0, SAFFRON)])
    )
    blue_sunglow: Palette = field(
        default_factory=lambda: Spreadable(
            [(0.0, BLUE), (0.5, TURQUOISE), (1.0, SUNGLOW)]
        )
    )
    margarita_red: Palette = field(
        default_factory=lambda: Spreadable(
            [(0.0, MARGARITA), (0.5, SUNGLOW), (1.0, RED_CRAYOLA)]
        )
    )
    yellow_fushia: Palette = field(
        default_factory=lambda: Spreadable(
            [(0.0, SUNGLOW), (0.5, RED_CRAYOLA), (1.0, FUSHIA)]
        )
    )


@dataclass
class _Diverging(ReversableMixin):
    onoff: Palette = field(
        default_factory=lambda: Spreadable(
            [
                (0.0, MARGARITA),
                (0.5, GHOST_WHITE),
                (1.0, RED_CRAYOLA),
            ]
        )
    )
    turquoise_blue: Palette = field(
        default_factory=lambda: Spreadable(
            [(0.0, TURQUOISE), (0.5, GHOST_WHITE), (1.0, BLUE)]
        )
    )
    turquoise_lightblue: Palette = field(
        default_factory=lambda: Spreadable(
            [
                (0.0, TURQUOISE),
                (0.5, GHOST_WHITE),
                (1.0, BLUE_CRAYOLA),
            ]
        )
    )
    red_turquoise: Palette = field(
        default_factory=lambda: Spreadable(
            [
                (0.0, RED_CRAYOLA),
                (0.5, GHOST_WHITE),
                (1.0, TURQUOISE),
            ]
        )
    )
    red_blue: Palette = field(
        default_factory=lambda: Spreadable(
            [(0.0, RED_CRAYOLA), (0.5, GHOST_WHITE), (1.0, BLUE)]
        )
    )
    red_lightblue: Palette = field(
        default_factory=lambda: Spreadable(
            [
                (0.0, RED_CRAYOLA),
                (0.5, GHOST_WHITE),
                (1.0, BLUE_CRAYOLA),
            ]
        )
    )
    saffron_blue_turquoise: Palette = field(
        default_factory=lambda: Spreadable(
            [
                (0.0, SAFFRON),
                (0.25, FUSHIA),
                (0.5, BLUE),
                (0.75, BLUE_CRAYOLA),
                (1.0, TURQUOISE),
            ]
        )
    )
    lightblue_margarita_red: Palette = field(
        default_factory=lambda: Spreadable(
            [
                (0.0, BLUE_CRAYOLA),
                (0.25, TURQUOISE),
                (0.5, MARGARITA),
                (0.75, SUNGLOW),
                (1.0, RED_CRAYOLA),
            ]
        )
    )


binary = _Binary()
qualitative = _Qualitative()
sequential = _Sequential()
diverging = _Diverging()
