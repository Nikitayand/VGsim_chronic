"""Сборка симуляции VGsim с модулем хронических пациентов.

Модуль ничего не меняет в движке: он только расставляет штатные параметры
``VGsim.Simulator`` так, чтобы внутри компартмента I образовались две
непересекающиеся субпопуляции — обычного и затяжного течения инфекции.

Порядок вызовов важен и зафиксирован в :meth:`ChronicModel.build`:

* ``set_population_size`` обнуляет распределение восприимчивых по группам,
  поэтому размеры демов задаются раньше ``set_susceptible``;
* ``set_population_size`` / ``set_susceptible`` / ``set_infectious`` движок
  запрещает вызывать после первого ``simulate()`` — модель нельзя досоздать
  на ходу, вся структура фиксируется до запуска;
* веса замен на служебном сайте задаются отдельно для каждого блока: VGsim
  выбрасывает из четвёрки весов текущий нуклеотид и требует, чтобы сумма
  оставшихся была положительной.
"""

from __future__ import annotations

import warnings

from .encoding import GenomeLayout, HostState
from .params import ModelConfig
from .theory import LinearisedEpidemic, linearise, mixing_matrix

__all__ = ["ChronicModel", "build_simulator"]

#: Выше этого числа гаплотипов имеет смысл включать memory_optimization движка.
_HAPLOTYPE_WARNING_THRESHOLD = 4096


class ChronicModel:
    """Конфигурация -> настроенный ``VGsim.Simulator``.

    >>> from vgsim_chronic import ChronicModel, default_config
    >>> model = ChronicModel(default_config())
    >>> simulator = model.build(seed=1234)      # doctest: +SKIP
    >>> simulator.simulate(10_000_000, epidemic_time=240)   # doctest: +SKIP
    """

    def __init__(self, config: ModelConfig, *, memory_optimization: bool = False):
        self.config = config
        self.memory_optimization = memory_optimization
        self.layout = GenomeLayout(genome_sites=config.mutations.genome_sites)
        self.immunity = config.immunity
        if (
            self.layout.haplotypes_number > _HAPLOTYPE_WARNING_THRESHOLD
            and not memory_optimization
        ):
            warnings.warn(
                f"пространство гаплотипов {self.layout.haplotypes_number}; "
                "служебный сайт увеличивает его вчетверо — рассмотрите "
                "memory_optimization=True",
                stacklevel=2,
            )

    # аналитика

    @property
    def linearised(self) -> LinearisedEpidemic:
        """Линеаризованная система для той же конфигурации."""
        return linearise(self.config)

    def check(self) -> list[str]:
        """Проверки, которые стоит пройти до запуска. Возвращает список замечаний."""
        problems: list[str] = []
        if not self.config.chronic.enabled:
            # переходов нет, блок затяжного течения недостижим: проверять нечего
            return problems
        linear = self.linearised
        r_chronic = linear.r0_chronic_block()
        if r_chronic >= 1.0:
            problems.append(
                f"R0 блока затяжного течения = {r_chronic:.3f} >= 1: линии затяжного "
                "течения самоподдерживаются, и наследование маркера перестаёт быть "
                "малой поправкой. Снизьте transmission_rate затяжного течения либо "
                "повысьте его recovery_rate."
            )
        elif r_chronic > 0.5:
            problems.append(
                f"R0 блока затяжного течения = {r_chronic:.3f}: поправка на "
                f"наследование маркера уже заметна "
                f"({linear.inheritance_overhead():.2f} лишних заражений на занос)."
            )
        module = self.config.chronic
        if module.chronic.clearance_rate >= module.acute.clearance_rate:
            problems.append(
                "затяжное течение выбывает не медленнее обычного — субпопуляция "
                "хронических пациентов не даёт удлинения инфекционного периода."
            )
        return problems

    # построение

    def build(self, seed: int | None = None, *, strict: bool = True):
        """Собрать и вернуть настроенный ``VGsim.Simulator``.

        :param seed: зерно генератора; ``None`` — случайное.
        :param strict: прервать сборку, если :meth:`check` нашёл нарушение
            условия ``R0`` затяжного блока < 1.
        """
        import VGsim  # импорт отложен: пакет можно использовать и без движка

        problems = self.check()
        for problem in problems:
            warnings.warn(problem, stacklevel=2)
        if strict and any("R0 блока затяжного течения" in p and ">= 1" in p for p in problems):
            raise ValueError(
                "модель не проходит проверку подкритичности затяжного блока; "
                "запуск с strict=False, если это сделано намеренно"
            )

        config = self.config
        population = config.population
        module = config.chronic
        layout = self.layout
        immunity = self.immunity

        simulator = VGsim.Simulator(
            number_of_sites=layout.total_sites,
            populations_number=population.n_demes,
            number_of_susceptible_groups=immunity.n_groups,
            seed=seed,
            memory_optimization=self.memory_optimization,
        )

        self._set_course_rates(simulator)
        self._set_mutations(simulator)
        self._set_immunity(simulator)
        self._set_demes(simulator)
        self._set_mixing(simulator)
        self._seed_epidemic(simulator)

        if config.npi is not None:
            simulator.set_npi(list(config.npi))

        return simulator

    # частями

    def _set_course_rates(self, simulator) -> None:
        """Интенсивности передачи, выбытия и секвенирования по блокам гаплотипов."""
        module = self.config.chronic
        layout, immunity = self.layout, self.immunity

        # Сначала параметры обычного течения — всем гаплотипам сразу, включая
        # недостижимые (маркер 'C'/'G'). Иначе у них остались бы заводские
        # значения движка, которые раздувают верхнюю оценку интенсивности
        # миграции и замедляют отбор с отклонением в GenerateMigration.
        acute = module.acute
        simulator.set_transmission_rate(acute.transmission_rate)
        simulator.set_recovery_rate(acute.recovery_rate)
        simulator.set_sampling_rate(acute.sampling_rate)
        simulator.set_susceptibility_type(immunity.recovered_group(HostState.ACUTE))

        chronic_block = layout.block(HostState.CHRONIC)
        chronic = module.chronic
        simulator.set_transmission_rate(chronic.transmission_rate, haplotype=chronic_block)
        simulator.set_recovery_rate(chronic.recovery_rate, haplotype=chronic_block)
        simulator.set_sampling_rate(chronic.sampling_rate, haplotype=chronic_block)
        simulator.set_susceptibility_type(
            immunity.recovered_group(HostState.CHRONIC), haplotype=chronic_block
        )

    def _set_mutations(self, simulator) -> None:
        """Настоящие мутации на сайтах генома и переходы течения на маркере."""
        layout = self.layout
        mutations = self.config.mutations
        module = self.config.chronic

        for site in layout.genome_site_indices:
            simulator.set_mutation_rate(mutations.rate, mutation=site)
            simulator.set_mutation_probabilities(list(mutations.weights), mutation=site)

        marker = layout.marker_site
        simulator.set_mutation_rate(0.0, mutation=marker)

        acute_block = layout.block(HostState.ACUTE)
        simulator.set_mutation_probabilities(
            layout.substitution_weights(HostState.CHRONIC),
            haplotype=acute_block,
            mutation=marker,
        )
        simulator.set_mutation_rate(
            module.progression_rate, haplotype=acute_block, mutation=marker
        )

        if module.reversion_rate > 0.0:
            chronic_block = layout.block(HostState.CHRONIC)
            simulator.set_mutation_probabilities(
                layout.substitution_weights(HostState.ACUTE),
                haplotype=chronic_block,
                mutation=marker,
            )
            simulator.set_mutation_rate(
                module.reversion_rate, haplotype=chronic_block, mutation=marker
            )

    def _set_immunity(self, simulator) -> None:
        """Восприимчивость по группам и потеря постинфекционного иммунитета.

        Восприимчивость намеренно не зависит от служебного сайта: маркер
        описывает течение болезни у хозяина, а не антигенные свойства
        патогена, и «узнавать» его иммунитет не должен.
        """
        immunity = self.immunity
        module = self.config.chronic

        for index, stratum in enumerate(self.config.population.strata):
            simulator.set_susceptibility(
                stratum.relative_susceptibility,
                susceptibility_type=immunity.naive_group(index),
            )

        for state in (HostState.ACUTE, HostState.CHRONIC):
            params = module.params_of(state)
            group = immunity.recovered_group(state)
            simulator.set_susceptibility(
                1.0 - params.immunity_protection, susceptibility_type=group
            )
            if params.waning_rate > 0.0:
                simulator.set_immunity_transition(
                    params.waning_rate, source=group, target=immunity.waning_target
                )

    def _set_demes(self, simulator) -> None:
        """Размеры демов, стартовое распределение по группам, контакты и надзор."""
        population = self.config.population
        immunity = self.immunity
        sizes = population.sizes

        for index, size in enumerate(sizes):
            simulator.set_population_size(size, population=index)

        for index, size in enumerate(sizes):
            group = immunity.naive_group(index)
            if group != 0:
                _move_susceptible(simulator, size, 0, group, index)

        for index, stratum in enumerate(population.strata):
            simulator.set_contact_density(stratum.contact_density, population=index)
            simulator.set_sampling_multiplier(stratum.sampling_multiplier, population=index)

    def _set_mixing(self, simulator) -> None:
        """Матрица контактов между демами."""
        population = self.config.population
        if population.n_demes < 2:
            return
        matrix = mixing_matrix(population.shares, population.assortativity)
        for source in range(population.n_demes):
            for target in range(population.n_demes):
                if source != target:
                    simulator.set_migration_probability(
                        float(matrix[source, target]), source=source, target=target
                    )

    def _seed_epidemic(self, simulator) -> None:
        """Занос инфекции обычного течения в дем 0.

        Движок в ``PrepareParameters`` сам создаёт одного заражённого нулевым
        гаплотипом в деме 0 (``FirstInfection``), а нулевой гаплотип в нашей
        раскладке — это как раз «обычное течение, дикий тип». Здесь только
        досеваются остальные заносы, если их заказано больше одного.
        """
        extra = self.config.population.initial_infected - 1
        if extra <= 0:
            return
        _seed_infectious(
            simulator,
            extra,
            self.immunity.naive_group(0),
            self.layout.haplotype(HostState.ACUTE, 0),
            0,
        )

    # отчёт

    def report(self) -> str:
        """Протокол конфигурации — то, что имеет смысл приводить в тексте работы."""
        config = self.config
        population = config.population
        module = config.chronic
        lines = [
            f"=== {config.label} ===",
            "",
            self.layout.describe(),
            "",
            "Клинические течения (1/сут):",
            f"  {'':<22}{'обычное':>12}{'затяжное':>12}",
            f"  {'передача beta':<22}{module.acute.transmission_rate:>12.4f}"
            f"{module.chronic.transmission_rate:>12.4f}",
            f"  {'выбытие gamma':<22}{module.acute.recovery_rate:>12.4f}"
            f"{module.chronic.recovery_rate:>12.4f}",
            f"  {'секвенирование':<22}{module.acute.sampling_rate:>12.4f}"
            f"{module.chronic.sampling_rate:>12.4f}",
            f"  {'средняя длительность':<22}{module.acute.mean_duration:>12.2f}"
            f"{module.chronic.mean_duration:>12.2f}",
            "",
            f"Прогрессирование обычное -> затяжное: sigma = {module.progression_rate:.5f} 1/сут",
            f"Доля заражений с затяжным течением  : {module.chronicity_risk:.3f}",
            f"Средний инфекционный период         : {module.mean_infectious_period:.2f} сут",
            "",
            "Страты хозяев (демы):",
            f"  {'имя':<12}{'размер':>12}{'доля':>8}{'CD':>7}{'надзор':>9}{'воспр.':>9}{'риск':>7}",
        ]
        for stratum, size in zip(population.strata, population.sizes):
            lines.append(
                f"  {stratum.name:<12}{size:>12d}{stratum.share:>8.3f}"
                f"{stratum.contact_density:>7.2f}{stratum.sampling_multiplier:>9.2f}"
                f"{stratum.relative_susceptibility:>9.2f}{stratum.chronicity_risk:>7.2f}"
            )
        lines += [
            f"  ассортативность контактов eps = {population.assortativity:.3f}",
            "",
            self.linearised.describe(),
        ]
        problems = self.check()
        if problems:
            lines += ["", "Замечания:"] + [f"  ! {p}" for p in problems]
        return "\n".join(lines)


def build_simulator(config: ModelConfig, seed: int | None = None, **kwargs):
    """Короткий путь: конфигурация -> готовый ``VGsim.Simulator``."""
    return ChronicModel(config, **kwargs).build(seed=seed)


def _move_susceptible(simulator, amount: int, source: int, target: int, deme: int) -> None:
    """Перевести восприимчивых между группами внутри дема."""
    try:
        simulator.set_susceptible(amount, source, target, population=deme)
        return
    except TypeError:
        pass

    import numpy as np

    susceptible = np.asarray(simulator.susceptible)
    if susceptible[deme, source] < amount:
        raise ValueError(
            f"в деме {deme} группе {source} не хватает восприимчивых: "
            f"{susceptible[deme, source]} < {amount}"
        )
    susceptible[deme, source] -= amount
    susceptible[deme, target] += amount


def _seed_infectious(
    simulator, amount: int, source_group: int, haplotype: int, deme: int
) -> None:
    """Перевести восприимчивых в инфицированные заданным гаплотипом."""
    try:
        simulator.set_infectious(amount, source_group, haplotype, population=deme)
        return
    except TypeError:
        pass

    import numpy as np

    susceptible = np.asarray(simulator.susceptible)
    infectious = np.asarray(simulator.infectious)
    if susceptible[deme, source_group] < amount:
        raise ValueError(
            f"в деме {deme} группе {source_group} не хватает восприимчивых для заноса"
        )
    susceptible[deme, source_group] -= amount
    infectious[deme, haplotype] += amount
