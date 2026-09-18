"""Аналитика модели: матрица контактов, R0, скорость роста, калибровка.

Здесь линеаризованная в окрестности безэпидемического состояния версия ровно
той динамики, которую разыгрывает VGsim. Она нужна по трём причинам:

1. чтобы сравнивать однородную и стратифицированную модели при **одинаковом**
   R0 — иначе разница в кривых объясняется просто разным R0, а не структурой;
2. чтобы контролировать главный дефект кодировки — наследование маркера
   при передаче (см. ``r0_chronic_block``);
3. чтобы численно показать, чем реализуемая в VGsim модель отличается от
   идеальной («эталонной»), в которой течение инфекции назначается заново
   при каждом заражении.

Обозначения соответствуют коду движка (``BirthRate`` и ``GenerateMigration``
в ``_BirthDeath.pyx``). Интенсивность заражений, идущих от носителя в деме p
к восприимчивым дема q, равна

    I[p, h] * S[q, g] * E[p, q] * bRate[h] * susceptibility[h, g],

где E — матрица эффективных контактов

    E[p, q] = sum_r m[p, r] * m[q, r] * contactDensity[r] / actualSize[r],

m — матрица «доля времени, проводимого жителем дема p в деме q»,
actualSize[r] = sum_p m[p, r] * size[p]. Формула одна и та же для p = q
(рождение внутри дема) и для p != q (принятая миграция).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import HostState
from .params import ModelConfig

__all__ = [
    "mixing_matrix",
    "actual_sizes",
    "effective_contact_matrix",
    "LinearisedEpidemic",
    "linearise",
    "calibrate_transmission",
]

N_STATES = 2

#  Матрица контактов


def mixing_matrix(shares, assortativity: float) -> np.ndarray:
    """Матрица ``m[p][q]`` — доля времени жителя дема p, проводимая в деме q.

    ``m[p][q] = (1 - eps) * w_q + eps * [p == q]``.

    При ``eps = 0`` каждый проводит в деме q долю времени, равную доле q в
    населении: это в точности однородное перемешивание. При ``eps = 1`` демы
    изолированы. Фактические размеры демов при любом eps совпадают с
    номинальными, поэтому VGsim не ругается на раздутые демы.
    """
    weights = np.asarray(shares, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("shares — непустой одномерный массив долей")
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    if not 0.0 <= assortativity <= 1.0:
        raise ValueError("assortativity должна лежать в [0, 1]")
    k = weights.size
    matrix = (1.0 - assortativity) * np.tile(weights, (k, 1))
    matrix += assortativity * np.eye(k)
    return matrix


def actual_sizes(mixing: np.ndarray, sizes) -> np.ndarray:
    """Сколько людей фактически находится в каждом деме: ``sum_p m[p][r] * N_p``."""
    return np.asarray(sizes, dtype=float) @ np.asarray(mixing, dtype=float)


def effective_contact_matrix(
    mixing: np.ndarray, sizes, contact_density
) -> np.ndarray:
    """Матрица эффективных контактов E из формулы в заголовке модуля."""
    mixing = np.asarray(mixing, dtype=float)
    density = np.asarray(contact_density, dtype=float)
    sizes_actual = actual_sizes(mixing, sizes)
    if np.any(sizes_actual <= 0):
        raise ValueError("фактический размер дема получился неположительным")
    return mixing @ np.diag(density / sizes_actual) @ mixing.T


#  Линеаризация


@dataclass(frozen=True)
class LinearisedEpidemic:
    """Линеаризованная в безэпидемическом состоянии система.

    Типы заражённых нумеруются как ``i = deme * 2 + course``, где course —
    значение :class:`~vgsim_chronic.encoding.HostState`.

    :param beta: интенсивности передачи по течениям, форма (2,).
    :param removal: выбытие ``gamma + sampling * multiplier`` по (дем, течение).
    :param progression: sigma, обычное -> затяжное.
    :param reversion: обратный переход, затяжное -> обычное.
    :param contact: матрица E, форма (демы, демы).
    :param susceptible_weight: ``N_q * relative_susceptibility_q``, форма (демы,).
    """

    beta: np.ndarray
    removal: np.ndarray
    progression: float
    reversion: float
    contact: np.ndarray
    susceptible_weight: np.ndarray

    # служебное

    @property
    def n_demes(self) -> int:
        return self.contact.shape[0]

    @property
    def size(self) -> int:
        return self.n_demes * N_STATES

    def _index(self, deme: int, course: int) -> int:
        return deme * N_STATES + course

    def transition_matrix(self) -> np.ndarray:
        """Матрица V: выбытие и переходы между течениями (без новых заражений)."""
        n = self.size
        matrix = np.zeros((n, n))
        for deme in range(self.n_demes):
            acute = self._index(deme, int(HostState.ACUTE))
            chronic = self._index(deme, int(HostState.CHRONIC))
            matrix[acute, acute] = self.removal[deme, 0] + self.progression
            matrix[chronic, chronic] = self.removal[deme, 1] + self.reversion
            matrix[chronic, acute] = -self.progression
            matrix[acute, chronic] = -self.reversion
        return matrix

    def infection_matrix(self, inherited: bool = True) -> np.ndarray:
        """Матрица F: ``F[i, j]`` — интенсивность порождения типом j заражений типа i.

        :param inherited: ``True`` — реализуемая в VGsim модель, где новый
            заражённый наследует течение донора вместе с гаплотипом;
            ``False`` — эталонная модель, где всякое заражение начинается
            с обычного течения.
        """
        n = self.size
        matrix = np.zeros((n, n))
        for source_deme in range(self.n_demes):
            for course in range(N_STATES):
                source = self._index(source_deme, course)
                for target_deme in range(self.n_demes):
                    rate = (
                        self.beta[course]
                        * self.contact[source_deme, target_deme]
                        * self.susceptible_weight[target_deme]
                    )
                    target_course = course if inherited else int(HostState.ACUTE)
                    matrix[self._index(target_deme, target_course), source] += rate
        return matrix

    # показатели

    def next_generation_matrix(self, inherited: bool = True) -> np.ndarray:
        """K = F * V^(-1) — матрица следующего поколения."""
        return self.infection_matrix(inherited) @ np.linalg.inv(self.transition_matrix())

    def r0(self, inherited: bool = True) -> float:
        """Базовое репродуктивное число — спектральный радиус K."""
        return _spectral_radius(self.next_generation_matrix(inherited))

    def growth_rate(self, inherited: bool = True) -> float:
        """Мальтузианская скорость роста r — старшее собственное число F - V."""
        matrix = self.infection_matrix(inherited) - self.transition_matrix()
        return float(np.max(np.linalg.eigvals(matrix).real))

    def r0_chronic_block(self) -> float:
        """R0 замкнутой на себя субпопуляции затяжного течения.

        Из-за наследования маркера линии затяжного течения размножаются
        самостоятельно, и матрица следующего поколения оказывается
        блочно-треугольной. Спектральный радиус блока «затяжное -> затяжное»
        обязан быть **меньше единицы**: только тогда такие линии подкритичны,
        сами эпидемию не поддерживают, а вносимая ими поправка ограничена
        величиной ``R_c / (1 - R_c)`` лишних заражений на один занос.
        """
        idx = [self._index(d, int(HostState.CHRONIC)) for d in range(self.n_demes)]
        big_f = self.infection_matrix(inherited=True)
        big_v = self.transition_matrix()
        sub_f = big_f[np.ix_(idx, idx)]
        sub_v = big_v[np.ix_(idx, idx)]
        return _spectral_radius(sub_f @ np.linalg.inv(sub_v))

    def r0_acute_block(self) -> float:
        """R0 блока обычного течения — то, что и разгоняет эпидемию."""
        idx = [self._index(d, int(HostState.ACUTE)) for d in range(self.n_demes)]
        big_f = self.infection_matrix(inherited=True)
        big_v = self.transition_matrix()
        sub_f = big_f[np.ix_(idx, idx)]
        sub_v = big_v[np.ix_(idx, idx)]
        return _spectral_radius(sub_f @ np.linalg.inv(sub_v))

    def inheritance_overhead(self) -> float:
        """Оценка сверху на завышение числа заражений из-за наследования маркера.

        Один «ошибочно затяжной» занос порождает подкритичную цепочку средним
        размером ``1 / (1 - R_c)``, то есть ``R_c / (1 - R_c)`` лишних заражений.
        """
        r_chronic = self.r0_chronic_block()
        if r_chronic >= 1.0:
            return float("inf")
        return r_chronic / (1.0 - r_chronic)

    def summary(self) -> dict[str, float]:
        return {
            "R0": self.r0(inherited=True),
            "R0_reference": self.r0(inherited=False),
            "R0_acute_block": self.r0_acute_block(),
            "R0_chronic_block": self.r0_chronic_block(),
            "growth_rate": self.growth_rate(inherited=True),
            "growth_rate_reference": self.growth_rate(inherited=False),
            "inheritance_overhead": self.inheritance_overhead(),
        }

    @property
    def module_enabled(self) -> bool:
        """Достижим ли блок затяжного течения."""
        return self.progression > 0.0 or self.reversion > 0.0

    def describe(self) -> str:
        s = self.summary()
        if not self.module_enabled:
            return "\n".join(
                [
                    f"R0                                   : {s['R0_acute_block']:.3f}",
                    f"скорость роста r, 1/сут              : "
                    f"{s['growth_rate_reference']:.4f}",
                    "модуль хронических пациентов выключен: блок затяжного "
                    "течения недостижим",
                ]
            )
        return "\n".join(
            [
                f"R0 (реализуемая в VGsim модель)      : {s['R0']:.3f}",
                f"R0 (эталонная модель без наследования): {s['R0_reference']:.3f}",
                f"  блок обычного течения               : {s['R0_acute_block']:.3f}",
                f"  блок затяжного течения              : {s['R0_chronic_block']:.3f}"
                + ("   <-- ДОЛЖЕН БЫТЬ < 1" if s["R0_chronic_block"] >= 1 else "  (< 1, норма)"),
                f"скорость роста r, 1/сут              : {s['growth_rate']:.4f}",
                f"  то же в эталонной модели            : {s['growth_rate_reference']:.4f}",
                f"избыток заражений на ошибочный занос : {s['inheritance_overhead']:.3f}",
            ]
        )

    def scaled(self, factor: float) -> "LinearisedEpidemic":
        """Копия с умноженными интенсивностями передачи."""
        return LinearisedEpidemic(
            beta=self.beta * factor,
            removal=self.removal,
            progression=self.progression,
            reversion=self.reversion,
            contact=self.contact,
            susceptible_weight=self.susceptible_weight,
        )


def _spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def linearise(config: ModelConfig) -> LinearisedEpidemic:
    """Построить линеаризованную систему по конфигурации модели."""
    population = config.population
    module = config.chronic
    sizes = np.asarray(population.sizes, dtype=float)
    density = np.array([s.contact_density for s in population.strata], dtype=float)
    multipliers = np.array([s.sampling_multiplier for s in population.strata], dtype=float)
    susceptibility = np.array(
        [s.relative_susceptibility for s in population.strata], dtype=float
    )

    mixing = mixing_matrix(population.shares, population.assortativity)
    contact = effective_contact_matrix(mixing, sizes, density)

    acute, chronic = module.acute, module.chronic
    removal = np.column_stack(
        [
            acute.recovery_rate + acute.sampling_rate * multipliers,
            chronic.recovery_rate + chronic.sampling_rate * multipliers,
        ]
    )
    return LinearisedEpidemic(
        beta=np.array([acute.transmission_rate, chronic.transmission_rate]),
        removal=removal,
        progression=module.progression_rate,
        reversion=module.reversion_rate,
        contact=contact,
        susceptible_weight=sizes * susceptibility,
    )


def calibrate_transmission(
    config: ModelConfig, target_r0: float, *, inherited: bool = True
) -> ModelConfig:
    """Сдвинуть интенсивности передачи так, чтобы R0 стал равен ``target_r0``.

    Матрица F линейна по beta, поэтому R0 однороден первой степени и
    достаточно одного деления — итерации не нужны. Соотношение
    ``beta_chronic / beta_acute`` при этом сохраняется, то есть сохраняется
    и содержательное допущение «затяжные больные заразны слабее».
    """
    if target_r0 <= 0:
        raise ValueError("target_r0 должен быть положительным")
    current = linearise(config).r0(inherited=inherited)
    if current <= 0:
        raise ValueError("нулевой R0: нечего масштабировать")
    return config.with_transmission_scaled(target_r0 / current)
