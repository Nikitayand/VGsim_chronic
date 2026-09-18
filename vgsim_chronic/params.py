"""Параметры модели: клинические течения, страты хозяев, иммунитет, мутации.

Все интенсивности заданы в единицах 1/сутки, все времена — в сутках.
Значения по умолчанию — заготовка порядка величины для SARS-CoV-2; числа,
которые пойдут в текст работы, должны быть заменены на выписанные из
источников главы 1.2 (поле ``source`` у каждого набора для того и заведено).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .encoding import HostState

__all__ = [
    "CourseParams",
    "ChronicModule",
    "MutationModel",
    "ImmunityModel",
    "Stratum",
    "HostPopulation",
    "ModelConfig",
    "progression_rate_from_risk",
    "chronicity_risk_from_strata",
    "DEFAULT_ACUTE",
    "DEFAULT_CHRONIC",
    "default_config",
    "homogeneous_config",
    "homogeneous_with_duration",
    "with_chronicity_risk",
]


#  Клиническое течение инфекции


@dataclass(frozen=True)
class CourseParams:
    """Интенсивности для одного клинического течения инфекции.

    :param transmission_rate: beta — интенсивность передачи, 1/сут.
    :param recovery_rate: gamma — интенсивность выбытия (выздоровление либо
        смерть; VGsim не различает эти исходы), 1/сут.
    :param sampling_rate: интенсивность секвенирования, 1/сут. В VGsim
        секвенирование **изымает** носителя из инфекционных, поэтому полное
        выбытие равно ``recovery_rate + sampling_rate * sampling_multiplier``.
    :param immunity_protection: доля, на которую снижена восприимчивость
        переболевшего этим течением (1.0 — полная защита, 0.0 — её нет).
    :param immunity_duration: среднее время жизни этой защиты, сут;
        ``None`` — защита пожизненная.
    :param source: ссылка на источник значений (идёт в протокол эксперимента).
    """

    transmission_rate: float
    recovery_rate: float
    sampling_rate: float
    immunity_protection: float = 0.9
    immunity_duration: float | None = 180.0
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("transmission_rate", "recovery_rate", "sampling_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} должно быть неотрицательным")
        if not 0.0 <= self.immunity_protection <= 1.0:
            raise ValueError("immunity_protection должно лежать в [0, 1]")
        if self.immunity_duration is not None and self.immunity_duration <= 0:
            raise ValueError("immunity_duration должно быть положительным")

    @property
    def clearance_rate(self) -> float:
        """Полное выбытие из инфекционных при множителе секвенирования 1."""
        return self.recovery_rate + self.sampling_rate

    @property
    def mean_duration(self) -> float:
        """Средняя длительность пребывания в этом течении, сут."""
        if self.clearance_rate == 0.0:
            return float("inf")
        return 1.0 / self.clearance_rate

    @property
    def waning_rate(self) -> float:
        """Интенсивность потери постинфекционного иммунитета, 1/сут."""
        return 0.0 if self.immunity_duration is None else 1.0 / self.immunity_duration

    def scaled(self, factor: float) -> "CourseParams":
        """Копия с умноженной интенсивностью передачи (нужна для калибровки R0)."""
        return replace(self, transmission_rate=self.transmission_rate * factor)


@dataclass(frozen=True)
class ChronicModule:
    """Собственно модуль хронических пациентов: два течения и переход между ними.

    :param acute: параметры обычного течения.
    :param chronic: параметры затяжного (хронического) течения.
    :param progression_rate: sigma — интенсивность перехода «обычное -> затяжное»,
        1/сут. В движке это событие мутации на служебном сайте.
    :param reversion_rate: интенсивность обратного перехода, 1/сут. По умолчанию
        нулевая: хроническое течение считается поглощающим до выбытия.
    """

    acute: CourseParams
    chronic: CourseParams
    progression_rate: float
    reversion_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.progression_rate < 0 or self.reversion_rate < 0:
            raise ValueError("интенсивности переходов должны быть неотрицательными")

    def params_of(self, state: HostState) -> CourseParams:
        return self.acute if state is HostState.ACUTE else self.chronic

    @property
    def enabled(self) -> bool:
        """Включён ли модуль. При нулевых переходах блок затяжного течения
        недостижим, и модель вырождается в обычную однокомпартментную."""
        return self.progression_rate > 0.0 or self.reversion_rate > 0.0

    @property
    def chronicity_risk(self) -> float:
        """Доля заражений, доходящих до затяжного течения (при множителе 1)."""
        total = self.acute.clearance_rate + self.progression_rate
        return 0.0 if total == 0 else self.progression_rate / total

    @property
    def mean_infectious_period(self) -> float:
        """Средняя длительность инфекции с учётом возможного затягивания, сут."""
        denominator = self.acute.clearance_rate + self.progression_rate
        if denominator == 0:
            return float("inf")
        return 1.0 / denominator + self.chronicity_risk * self.chronic.mean_duration

    @property
    def mean_generation_time(self) -> float:
        """Среднее время от заражения до передачи дальше, сут.

        Считается как первый момент профиля заразности, взвешенный самой
        заразностью::

            Tg = [b_a·Da² + q·b_c·(Da·Dc + Dc²)] / [b_a·Da + q·b_c·Dc],

        где Da = 1/(gamma_a + s_a + sigma) — среднее время в обычном течении,
        Dc — в затяжном, q — доля дошедших до затяжного течения.

        Величина проверяемая: для SARS-CoV-2 интервал генерации известен из
        независимых исследований. Она же отделяет два способа получить
        медленный хвост — растянуть инфекционный период у всех (тогда Tg
        растёт пропорционально) или вынести затяжные случаи в малую слабо
        передающую субпопуляцию (тогда Tg почти не меняется).
        """
        acute_out = self.acute.clearance_rate + self.progression_rate
        if acute_out == 0:
            return float("inf")
        acute_time = 1.0 / acute_out
        risk = self.chronicity_risk
        chronic_time = self.chronic.mean_duration
        beta_a, beta_c = self.acute.transmission_rate, self.chronic.transmission_rate
        weight = beta_a * acute_time + risk * beta_c * chronic_time
        if weight == 0:
            return float("nan")
        moment = beta_a * acute_time**2 + risk * beta_c * (
            acute_time * chronic_time + chronic_time**2
        )
        return moment / weight

    def scaled(self, factor: float) -> "ChronicModule":
        return replace(
            self, acute=self.acute.scaled(factor), chronic=self.chronic.scaled(factor)
        )


def progression_rate_from_risk(risk: float, acute_clearance_rate: float) -> float:
    """Подобрать sigma так, чтобы доля затяжных случаев равнялась ``risk``.

    Носитель покидает обычное течение с интенсивностью ``sigma + gamma_a + s_a``,
    и доля ушедших в затяжное течение равна ``sigma / (sigma + gamma_a + s_a)``.
    Отсюда ``sigma = risk * (gamma_a + s_a) / (1 - risk)``.
    """
    if not 0.0 <= risk < 1.0:
        raise ValueError("risk должен лежать в [0, 1)")
    return risk * acute_clearance_rate / (1.0 - risk)


def chronicity_risk_from_strata(strata: "tuple[Stratum, ...] | list[Stratum]") -> float:
    """Средневзвешенный по населению риск затяжного течения.

    Так медицинское обоснование главы 1.2 (доля коморбидных в популяции и
    повышенный у них риск затяжного выделения вируса) превращается в один
    параметр sigma движка. Разные sigma по демам VGsim задать не позволяет:
    массив ``mRate`` проиндексирован гаплотипом и сайтом, но не популяцией.
    """
    total_share = sum(s.share for s in strata)
    if total_share <= 0:
        raise ValueError("суммарная доля страт должна быть положительной")
    return sum(s.share * s.chronicity_risk for s in strata) / total_share


#  Эволюция генома


@dataclass(frozen=True)
class MutationModel:
    """Мутации на настоящих (не служебных) сайтах генома.

    :param genome_sites: сколько изменчивых сайтов моделируется.
    :param rate: интенсивность мутирования одного сайта, 1/сут.
    :param weights: веса замен в порядке ATCG.
    """

    genome_sites: int = 2
    rate: float = 3e-5
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.genome_sites < 0:
            raise ValueError("genome_sites должно быть неотрицательным")
        if self.rate < 0:
            raise ValueError("rate должно быть неотрицательным")
        if len(self.weights) != 4:
            raise ValueError("weights — четыре веса в порядке ATCG")


#  Страты хозяев (демы)


@dataclass(frozen=True)
class Stratum:
    """Постоянная страта хозяев, реализуемая отдельным демом VGsim.

    Дем в VGsim — пожизненный признак: при событии MIGRATION заражается
    восприимчивый из целевого дема, сам носитель никуда не переезжает. Для
    признака «наличие хронического заболевания» это ровно та семантика,
    которая нужна.

    :param name: имя страты для подписей и протокола.
    :param share: доля страты в населении.
    :param chronicity_risk: вероятность затяжного течения у члена страты.
        В сам движок по-стратно не передаётся (см. ``chronicity_risk_from_strata``),
        используется для вывода общего sigma и для отчёта.
    :param contact_density: плотность контактов **в этом деме**. Множитель
        относится к месту встречи, а не к человеку: житель другого дема,
        оказавшийся здесь, получает ту же плотность.
    :param sampling_multiplier: множитель интенсивности секвенирования для
        жителей дема — им моделируется более плотное тестирование и
        госпитализация коморбидных пациентов. Внимание: секвенирование в VGsim
        изымает носителя, поэтому множитель влияет и на длительность инфекции.
    :param relative_susceptibility: относительная восприимчивость наивных
        жителей страты. Действует только до первого перенесённого заражения:
        группу иммунитета после выздоровления движок выбирает по гаплотипу
        (``suscType``), а не по дему, поэтому переболевшие всех страт попадают
        в общую группу. Для стойкого эффекта используйте ``contact_density``.
    """

    name: str
    share: float
    chronicity_risk: float = 0.0
    contact_density: float = 1.0
    sampling_multiplier: float = 1.0
    relative_susceptibility: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.share <= 1.0:
            raise ValueError(f"страта {self.name!r}: share должна лежать в (0, 1]")
        if not 0.0 <= self.chronicity_risk < 1.0:
            raise ValueError(f"страта {self.name!r}: chronicity_risk должен лежать в [0, 1)")
        for name in ("contact_density", "sampling_multiplier", "relative_susceptibility"):
            if getattr(self, name) < 0:
                raise ValueError(f"страта {self.name!r}: {name} должно быть неотрицательным")


@dataclass(frozen=True)
class HostPopulation:
    """Население: размер, разбиение на страты и то, как страты перемешиваются.

    :param size: полное число хозяев.
    :param strata: страты; их доли нормируются на единицу.
    :param assortativity: ассортативность контактов ``eps`` из [0, 1].
        Матрица контактов строится как ``m[p][q] = (1 - eps) * w_q + eps * [p == q]``.
        При ``eps = 0`` перемешивание пропорционально численности, и модель
        в точности вырождается в однородную (проверяется тестом);
        ``eps = 1`` — полностью изолированные демы. Такая параметризация
        сохраняет «фактические размеры» демов при любом eps, поэтому
        предупреждение VGsim о раздутых демах не возникает.
    :param initial_infected: сколько заражённых обычного течения занести в дем 0.
    """

    size: int
    strata: tuple[Stratum, ...]
    assortativity: float = 0.15
    initial_infected: int = 1

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size должен быть положительным")
        if not self.strata:
            raise ValueError("нужна хотя бы одна страта")
        if not 0.0 <= self.assortativity <= 1.0:
            raise ValueError("assortativity должна лежать в [0, 1]")
        if self.initial_infected < 1:
            raise ValueError("initial_infected должно быть не меньше 1")
        names = [s.name for s in self.strata]
        if len(set(names)) != len(names):
            raise ValueError("имена страт должны быть различны")

    @property
    def shares(self) -> tuple[float, ...]:
        total = sum(s.share for s in self.strata)
        return tuple(s.share / total for s in self.strata)

    @property
    def sizes(self) -> tuple[int, ...]:
        """Целые размеры демов, в сумме дающие ``size``."""
        raw = [int(round(self.size * w)) for w in self.shares]
        raw[0] += self.size - sum(raw)
        if min(raw) <= 0:
            raise ValueError("слишком мелкая страта: размер дема получился нулевым")
        return tuple(raw)

    @property
    def n_demes(self) -> int:
        return len(self.strata)


#  Иммунитет


@dataclass(frozen=True)
class ImmunityModel:
    """Раскладка групп восприимчивости.

    Группы нумеруются так::

        0 .. K-1     наивные жители страты k
        K + j        переболевшие течением j (0 — обычное, 1 — затяжное)

    :param waning_target: группа, в которую переходят переболевшие при потере
        иммунитета. Переход сохраняет дем, но не страту: ``set_immunity_transition``
        задаёт общие для всех демов пары «источник -> цель».
    """

    n_strata: int
    n_states: int = 2
    waning_target: int = 0

    @property
    def n_groups(self) -> int:
        return self.n_strata + self.n_states

    def naive_group(self, stratum_index: int) -> int:
        if not 0 <= stratum_index < self.n_strata:
            raise IndexError("нет такой страты")
        return stratum_index

    def recovered_group(self, state: HostState) -> int:
        return self.n_strata + int(state)


#  Полная конфигурация


@dataclass(frozen=True)
class ModelConfig:
    """Всё, что нужно, чтобы собрать симуляцию."""

    population: HostPopulation
    chronic: ChronicModule
    mutations: MutationModel = field(default_factory=MutationModel)
    npi: tuple[float, float, float] | None = None
    label: str = "стратифицированная модель"

    @property
    def immunity(self) -> ImmunityModel:
        return ImmunityModel(n_strata=self.population.n_demes)

    def with_transmission_scaled(self, factor: float) -> "ModelConfig":
        return replace(self, chronic=self.chronic.scaled(factor))


#  Значения по умолчанию

#: Обычное течение SARS-CoV-2: инфекционный период порядка 8-9 сут.
#:
#: Опора: Shanghai Omicron cohort (n = 198 262) даёт медиану времени выделения
#: вируса 8,29 сут (IQR 5,33-11,27); у пациентов без коморбидности 8,20 сут,
#: с коморбидностью 9,29 сут. Chen et al., J Infect Public Health, 2022;
#: https://doi.org/10.1016/j.jiph.2022.11.017
DEFAULT_ACUTE = CourseParams(
    transmission_rate=0.30,
    recovery_rate=1 / 9,
    sampling_rate=0.004,
    immunity_protection=0.90,
    immunity_duration=None,
    source="Chen et al., J Infect Public Health 2022 (Shanghai Omicron, n=198262)",
)

#: Затяжное течение: выделение вируса месяцами, но передача подавлена
#: наблюдением и изоляцией, а надзор за такими пациентами плотнее.
#:
#: Опора: ONS-CIS (Великобритания, 90 146 человек) — доля инфекций, длящихся
#: 30 сут и более, оценена в 0,7-3,5 %, 60 сут и более — в 0,1-0,5 %.
#: Отсюда условное распределение остатка: P(>=60 | >=30) ~ 0,14, то есть
#: сверх тридцати суток инфекция длится в среднем ещё около 15 сут, а полная
#: длительность затяжного случая - порядка 45 сут. Ghafari et al., Nature 626,
#: 2024; https://doi.org/10.1038/s41586-024-07029-4
#:
#: Интенсивность передачи здесь не свободна: условие подкритичности блока,
#: ``beta_c < 1 / mean_duration_c``, требует ``beta_c < 0.028``. Содержательно
#: это ограничение означает «затяжные носители не образуют самоподдерживающийся
#: резервуар» — для SARS-CoV-2 это выполняется, поскольку такие пациенты
#: находятся под наблюдением или в стационаре.
DEFAULT_CHRONIC = CourseParams(
    transmission_rate=0.006,
    recovery_rate=1 / 45,
    sampling_rate=0.006,
    immunity_protection=0.80,
    immunity_duration=None,
    source="Ghafari et al., Nature 2024 (ONS-CIS, персистирующие инфекции)",
)


def default_config(
    size: int = 2_000_000,
    comorbid_share: float = 0.17,
    comorbid_chronicity_risk: float = 0.065,
    general_chronicity_risk: float = 0.012,
    assortativity: float = 0.15,
    genome_sites: int = 2,
    immunity_duration: float | None = None,
) -> ModelConfig:
    """Стратифицированная модель: общая популяция плюс страта коморбидных.

    Доли по умолчанию опираются на шанхайскую когорту Omicron: коморбидность
    зарегистрирована у 33 896 из 198 262 заболевших, то есть примерно у 17 %.
    Риски затяжного течения подобраны так, чтобы средневзвешенный по населению
    риск составил около 2 % — середина интервала 0,7-3,5 %, полученного по
    ONS-CIS для инфекций длительностью 30 сут и более. Разбиение общего риска
    между стратами (примерно пятикратное превышение у коморбидных) — модельное
    допущение, которое в тексте работы нужно снабдить собственной ссылкой.

    :param immunity_duration: срок жизни постинфекционного иммунитета, сут.
        По умолчанию ``None`` — иммунитет держится весь рассматриваемый
        горизонт: так сравнение однородной и стратифицированной моделей
        касается одной волны и её спада, без наложения повторной волны.
    """
    strata = (
        Stratum(
            name="general",
            share=1.0 - comorbid_share,
            chronicity_risk=general_chronicity_risk,
        ),
        Stratum(
            name="comorbid",
            share=comorbid_share,
            chronicity_risk=comorbid_chronicity_risk,
            sampling_multiplier=3.0,
            relative_susceptibility=1.2,
        ),
    )
    risk = chronicity_risk_from_strata(strata)
    acute = replace(DEFAULT_ACUTE, immunity_duration=immunity_duration)
    chronic = replace(DEFAULT_CHRONIC, immunity_duration=immunity_duration)
    module = ChronicModule(
        acute=acute,
        chronic=chronic,
        progression_rate=progression_rate_from_risk(risk, acute.clearance_rate),
    )
    return ModelConfig(
        population=HostPopulation(size=size, strata=strata, assortativity=assortativity),
        chronic=module,
        mutations=MutationModel(genome_sites=genome_sites),
        label="стратифицированная модель",
    )


def homogeneous_config(
    reference: ModelConfig,
    *,
    label: str = "однородная модель",
) -> ModelConfig:
    """Однородная модель сравнения: один дем, одно течение инфекции.

    Строится не «на глаз», а по правилу: тот же размер населения, то же
    среднее время инфекционности и (после ``calibrate``) тот же R0. Это
    и есть модель, которую подогнал бы эпидемиолог, не знающий про
    хронических пациентов.
    """
    module = reference.chronic
    mean_period = module.mean_infectious_period
    clearance = 1.0 / mean_period
    # доля секвенируемых сохраняется, чтобы выборки были сопоставимы по объёму
    sampling_share = _sampling_share(module)
    acute = CourseParams(
        transmission_rate=module.acute.transmission_rate,
        recovery_rate=clearance * (1.0 - sampling_share),
        sampling_rate=clearance * sampling_share,
        immunity_protection=module.acute.immunity_protection,
        immunity_duration=module.acute.immunity_duration,
        source="производный набор: усреднение стратифицированной модели",
    )
    flat = ChronicModule(acute=acute, chronic=acute, progression_rate=0.0)
    stratum = Stratum(name="all", share=1.0)
    return ModelConfig(
        population=HostPopulation(
            size=reference.population.size,
            strata=(stratum,),
            assortativity=0.0,
            initial_infected=reference.population.initial_infected,
        ),
        chronic=flat,
        mutations=reference.mutations,
        npi=reference.npi,
        label=label,
    )


def with_chronicity_risk(config: ModelConfig, risk: float) -> ModelConfig:
    """Копия конфигурации с другой долей заражений, доходящих до затяжного течения.

    Длительности обоих течений при этом не меняются: варьируется только вес
    затяжной компоненты в смеси. Нужно для главы 3.3, где эта доля оценивается
    по данным и сравнивается с литературной.
    """
    module = config.chronic
    sigma = progression_rate_from_risk(risk, module.acute.clearance_rate)
    return replace(config, chronic=replace(module, progression_rate=sigma))


def homogeneous_with_duration(
    reference: ModelConfig, mean_days: float, *, label: str | None = None
) -> ModelConfig:
    """Однородная модель с заданным средним инфекционным периодом.

    Свободный аналог доли затяжных случаев: у однородной модели единственная
    ручка формы, помимо R0, — это её единственный временной масштаб.
    """
    if mean_days <= 0:
        raise ValueError("mean_days должно быть положительным")
    base = homogeneous_config(reference)
    clearance = 1.0 / mean_days
    share = _sampling_share(reference.chronic)
    acute = replace(
        base.chronic.acute,
        recovery_rate=clearance * (1.0 - share),
        sampling_rate=clearance * share,
    )
    module = ChronicModule(acute=acute, chronic=acute, progression_rate=0.0)
    return replace(
        base, chronic=module, label=label or f"однородная, D={mean_days:.4g} сут"
    )


def _sampling_share(module: ChronicModule) -> float:
    """Ожидаемая доля заражений, попавших в выборку, в стратифицированной модели."""
    risk = module.chronicity_risk
    acute_total = module.acute.clearance_rate + module.progression_rate
    if acute_total == 0:
        return 0.0
    share_acute = (1.0 - risk) * module.acute.sampling_rate / module.acute.clearance_rate
    share_chronic = 0.0
    if module.chronic.clearance_rate > 0:
        share_chronic = risk * module.chronic.sampling_rate / module.chronic.clearance_rate
    return min(1.0, share_acute + share_chronic)
