"""Кодировка субпопуляции хронических пациентов внутри компартмента I.

Пространство состояний заражённого хозяина в VGsim — это пара (дем, гаплотип).
Никакой третьей координаты у движка нет: массивы ``bRate``, ``dRate``, ``sRate``
проиндексированы только гаплотипом, а по дему параметризуются лишь размер
популяции, плотность контактов, множитель секвенирования и матрица контактов.

Поэтому «клиническое течение инфекции» (обычное или затяжное/хроническое)
записывается в уже существующую координату — в геном, в один служебный сайт.
Это и есть выделение субпопуляции без добавления нового компартмента: множество
инфицированных I разбивается на два непересекающихся блока гаплотипов, каждому
из которых движок позволяет назначить собственные интенсивности передачи,
выздоровления и секвенирования.

Соглашение о нумерации гаплотипов в VGsim (см. ``calculate_string_from_haplotype``
в ``_BirthDeath.pyx``): алфавит "ATCG" -> цифры 0..3, сайт 0 — старший разряд.
Служебный сайт ставится нулевым, поэтому индексы разбиваются на непрерывные
блоки::

    haplotype = state_digit * 4**genome_sites + genome_index

Блок ACUTE  (маркер 'A') занимает индексы [0, B),
блок CHRONIC (маркер 'T') — индексы [B, 2B), где B = 4**genome_sites.
Состояния маркера 'C' и 'G' недостижимы: переходы в них запрещены нулевыми
весами замен, а начальный занос всегда происходит в блок ACUTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

NUCLEOTIDES = "ATCG"


class HostState(IntEnum):
    """Клиническое течение инфекции у носителя.

    Значение перечисления — это цифра служебного сайта в нумерации VGsim.
    """

    ACUTE = 0  # 'A' — обычное течение
    CHRONIC = 1  # 'T' — затяжное (хроническое) течение

    @property
    def nucleotide(self) -> str:
        return NUCLEOTIDES[int(self)]

    @property
    def title(self) -> str:
        return {HostState.ACUTE: "обычное течение", HostState.CHRONIC: "затяжное течение"}[self]


@dataclass(frozen=True)
class GenomeLayout:
    """Раскладка генома: служебный сайт-маркер плюс настоящие изменчивые сайты.

    :param genome_sites: число настоящих (эволюционирующих) сайтов генома.
    :param marker_site: индекс служебного сайта; всегда 0 — старший разряд,
        чтобы блоки состояний были непрерывными диапазонами индексов.
    """

    genome_sites: int = 2
    marker_site: int = 0

    def __post_init__(self) -> None:
        if self.genome_sites < 0:
            raise ValueError("genome_sites должно быть неотрицательным")
        if self.marker_site != 0:
            raise ValueError(
                "маркер обязан быть сайтом 0: только тогда блоки состояний — "
                "непрерывные диапазоны индексов гаплотипов"
            )

    # размеры

    @property
    def total_sites(self) -> int:
        """Сколько сайтов заказывать у ``VGsim.Simulator``."""
        return self.genome_sites + 1

    @property
    def block_size(self) -> int:
        """Число гаплотипов внутри одного клинического состояния."""
        return 4 ** self.genome_sites

    @property
    def haplotypes_number(self) -> int:
        """Полный размер пространства гаплотипов, который заводит движок."""
        return 4 ** self.total_sites

    @property
    def genome_site_indices(self) -> list[int]:
        """Индексы настоящих сайтов генома в нумерации VGsim."""
        return list(range(1, self.total_sites))

    # отображения

    def haplotype(self, state: HostState, genome_index: int = 0) -> int:
        """Индекс гаплотипа по клиническому состоянию и геному."""
        if not 0 <= genome_index < self.block_size:
            raise IndexError(f"genome_index вне диапазона [0, {self.block_size})")
        return int(state) * self.block_size + genome_index

    def block(self, state: HostState) -> list[int]:
        """Все индексы гаплотипов данного клинического состояния."""
        start = int(state) * self.block_size
        return list(range(start, start + self.block_size))

    def split(self, haplotype: int) -> tuple[int, int]:
        """Разложить индекс гаплотипа на (цифра маркера, индекс генома)."""
        if not 0 <= haplotype < self.haplotypes_number:
            raise IndexError("гаплотип вне диапазона")
        return divmod(haplotype, self.block_size)

    def state_of(self, haplotype: int) -> HostState | None:
        """Клиническое состояние носителя; ``None`` для недостижимых гаплотипов."""
        digit, _ = self.split(haplotype)
        try:
            return HostState(digit)
        except ValueError:
            return None

    def genome_of(self, haplotype: int) -> int:
        """Настоящая часть генома — то, что и есть биологический гаплотип."""
        return self.split(haplotype)[1]

    def unreachable_haplotypes(self) -> list[int]:
        """Гаплотипы с маркером 'C'/'G': движок их заводит, модель не использует."""
        return list(range(2 * self.block_size, self.haplotypes_number))

    # строковое

    def to_string(self, haplotype: int) -> str:
        """Строковое представление в нотации VGsim, например 'TAC'."""
        digits = []
        value = haplotype
        for _ in range(self.total_sites):
            digits.append(NUCLEOTIDES[value % 4])
            value //= 4
        return "".join(reversed(digits))

    def genome_string(self, haplotype: int) -> str:
        """То же, но без служебного сайта — настоящий гаплотип патогена."""
        return self.to_string(haplotype)[1:]

    def pattern(self, state: HostState) -> str:
        """Шаблон блока с подстановочными знаками, например 'A**'."""
        return state.nucleotide + "*" * self.genome_sites

    def substitution_weights(self, target: HostState) -> list[float]:
        """Веса замен на служебном сайте, разрешающие только переход в ``target``.

        VGsim принимает список из четырёх весов в порядке ATCG и выбрасывает из
        него элемент, соответствующий текущему нуклеотиду. Единица напротив
        целевого состояния и нули в остальных позициях делают переход
        детерминированным по направлению.
        """
        weights = [0.0, 0.0, 0.0, 0.0]
        weights[int(target)] = 1.0
        return weights

    def describe(self) -> str:
        """Человекочитаемое описание раскладки — для протокола эксперимента."""
        lines = [
            f"Сайтов в модели: {self.total_sites} "
            f"(служебный маркер + {self.genome_sites} изменчивых)",
            f"Гаплотипов у движка: {self.haplotypes_number}, "
            f"из них используется {2 * self.block_size}",
        ]
        for state in (HostState.ACUTE, HostState.CHRONIC):
            block = self.block(state)
            lines.append(
                f"  {state.name:<7} ({state.title}): маркер '{state.nucleotide}', "
                f"шаблон {self.pattern(state)}, индексы {block[0]}..{block[-1]}"
            )
        return "\n".join(lines)
