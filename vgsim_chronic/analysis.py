"""Разбор результатов симуляции: траектории, выборка, генеалогия, диагностика.

Траектории собираются не методом ``get_data_infectious`` движка (он делает
отдельный проход по всей цепочке событий для каждой пары «дем, гаплотип», а
служебный сайт умножает число гаплотипов вчетверо), а одним разбором
выгруженной цепочки событий средствами numpy.

Семантика событий взята из ``events.pxi`` и обработчиков в ``_BirthDeath.pyx``::

    BIRTH      (t, 0, h,     p,   si,  hapNum)  -> +1 к (p, h)
    DEATH      (t, 1, h,     p,   si,  0)       -> -1 к (p, h)
    SAMPLING   (t, 2, h,     p,   si,  0)       -> -1 к (p, h), +1 к выборке
    MUTATION   (t, 3, h_old, p,   h_new, 0)     -> перенос между гаплотипами
    SUSCCHANGE (t, 4, g_old, p,   g_new, 0)     -> только восприимчивые
    MIGRATION  (t, 5, h,     p_s, si,  p_t)     -> +1 к (p_t, h)

Событие MUTATION на служебном сайте — это и есть переход «обычное течение ->
затяжное»: гаплотип меняет старший разряд, носитель переходит в другую
субпопуляцию, оставаясь в своём деме.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .encoding import GenomeLayout, HostState

__all__ = [
    "EventChain",
    "Trajectories",
    "EpidemicSummary",
    "load_event_chain",
    "collect_trajectories",
    "summarise",
    "inheritance_diagnostics",
    "sample_table",
    "tip_annotations",
    "terminal_branch_lengths",
]

BIRTH, DEATH, SAMPLING, MUTATION, SUSCCHANGE, MIGRATION, MULTITYPE = range(7)


#  Цепочка событий


@dataclass(frozen=True)
class EventChain:
    """Полная история симуляции в виде массивов numpy."""

    time: np.ndarray
    kind: np.ndarray
    haplotype: np.ndarray
    population: np.ndarray
    new_haplotype: np.ndarray
    new_population: np.ndarray

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        return float(self.time[-1]) if len(self) else 0.0

    def counts(self) -> dict[str, int]:
        names = {
            BIRTH: "births",
            DEATH: "recoveries",
            SAMPLING: "samples",
            MUTATION: "mutations",
            SUSCCHANGE: "immunity_transitions",
            MIGRATION: "migrations",
        }
        totals = np.bincount(self.kind, minlength=MULTITYPE + 1)
        return {name: int(totals[code]) for code, name in names.items()}


def load_event_chain(simulator, path: str | None = None) -> EventChain:
    """Выгрузить цепочку событий из симулятора.

    Штатный ``export_chain_events`` умеет только писать ``.npy``, поэтому файл
    создаётся во временном каталоге и сразу удаляется, если путь не задан явно.

    Хвост массива обрезается по первому нарушению строгой монотонности времени:
    так отсекаются и нули за концом записи, и следы неудачной попытки, которую
    движок отбросил через ``Restart`` (он сбрасывает указатель, но не чистит
    массивы).
    """
    cleanup = path is None
    if path is None:
        handle, path = tempfile.mkstemp(suffix=".npy")
        os.close(handle)
    stem = path[:-4] if path.endswith(".npy") else path
    try:
        simulator.export_chain_events(stem)
        raw = np.load(stem + ".npy", allow_pickle=False)
    finally:
        if cleanup and os.path.exists(stem + ".npy"):
            os.remove(stem + ".npy")

    time = np.asarray(raw[0], dtype=float)
    valid = _valid_prefix(time)
    return EventChain(
        time=time[:valid],
        kind=np.asarray(raw[1][:valid], dtype=np.int64),
        haplotype=np.asarray(raw[2][:valid], dtype=np.int64),
        population=np.asarray(raw[3][:valid], dtype=np.int64),
        new_haplotype=np.asarray(raw[4][:valid], dtype=np.int64),
        new_population=np.asarray(raw[5][:valid], dtype=np.int64),
    )


def _valid_prefix(time: np.ndarray) -> int:
    if time.size == 0:
        return 0
    steps = np.diff(time)
    broken = np.nonzero(steps <= 0.0)[0]
    return int(broken[0] + 1) if broken.size else int(time.size)


#  Траектории


@dataclass(frozen=True)
class Trajectories:
    """Численности во времени, разложенные по демам и клиническим течениям.

    :param time: сетка времени, сут, форма (T,).
    :param infectious: текущее число заражённых, форма (демы, течения, T).
    :param sampled: накопленное число секвенированных, форма (демы, течения, T).
    :param incidence: накопленное число новых заражений, форма (демы, течения, T).
    """

    time: np.ndarray
    infectious: np.ndarray
    sampled: np.ndarray
    incidence: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.infectious.sum(axis=(0, 1))

    def by_state(self, state: HostState) -> np.ndarray:
        return self.infectious[:, int(state), :].sum(axis=0)

    def by_deme(self, deme: int) -> np.ndarray:
        return self.infectious[deme].sum(axis=0)

    @property
    def chronic_load(self) -> np.ndarray:
        """Число носителей затяжного течения — прокси затяжной нагрузки на койки."""
        return self.by_state(HostState.CHRONIC)

    @property
    def chronic_share(self) -> np.ndarray:
        total = self.total
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, self.chronic_load / np.maximum(total, 1), 0.0)
        return share


def collect_trajectories(
    simulator,
    layout: GenomeLayout,
    *,
    steps: int = 400,
    chain: EventChain | None = None,
    initial_infectious: np.ndarray | None = None,
) -> Trajectories:
    """Восстановить траектории из цепочки событий.

    :param steps: число точек сетки времени.
    :param chain: уже загруженная цепочка (чтобы не выгружать её дважды).
    :param initial_infectious: начальные заражённые по (дем, течение);
        по умолчанию один заражённый обычного течения в деме 0 — так
        засевает эпидемию сам движок.
    """
    chain = chain if chain is not None else load_event_chain(simulator)
    n_demes = int(simulator.populations_number)
    n_states = 2
    n_cells = n_demes * n_states

    state_of = _state_lookup(layout)

    def cell(population: np.ndarray, haplotype: np.ndarray) -> np.ndarray:
        return population * n_states + state_of[haplotype]

    times: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    deltas: list[np.ndarray] = []

    def push(mask: np.ndarray, population: np.ndarray, haplotype: np.ndarray, sign: int):
        if not mask.any():
            return
        times.append(chain.time[mask])
        cells.append(cell(population[mask], haplotype[mask]))
        deltas.append(np.full(int(mask.sum()), sign, dtype=np.int64))

    is_birth = chain.kind == BIRTH
    is_removal = (chain.kind == DEATH) | (chain.kind == SAMPLING)
    is_mutation = chain.kind == MUTATION
    is_migration = chain.kind == MIGRATION
    is_sampling = chain.kind == SAMPLING

    push(is_birth, chain.population, chain.haplotype, +1)
    push(is_removal, chain.population, chain.haplotype, -1)
    push(is_mutation, chain.population, chain.haplotype, -1)
    push(is_mutation, chain.population, chain.new_haplotype, +1)
    push(is_migration, chain.new_population, chain.haplotype, +1)

    grid = np.linspace(0.0, chain.duration, steps + 1) if len(chain) else np.zeros(1)
    infectious = _accumulate(times, cells, deltas, grid, n_cells)
    if initial_infectious is None:
        infectious[0] += 1.0  # дем 0, обычное течение: занос от FirstInfection
    else:
        infectious += np.asarray(initial_infectious, dtype=float).reshape(-1, 1)

    sampled = _accumulate(
        [chain.time[is_sampling]],
        [cell(chain.population[is_sampling], chain.haplotype[is_sampling])],
        [np.ones(int(is_sampling.sum()), dtype=np.int64)],
        grid,
        n_cells,
    )
    incidence = _accumulate(
        [chain.time[is_birth], chain.time[is_migration]],
        [
            cell(chain.population[is_birth], chain.haplotype[is_birth]),
            cell(chain.new_population[is_migration], chain.haplotype[is_migration]),
        ],
        [
            np.ones(int(is_birth.sum()), dtype=np.int64),
            np.ones(int(is_migration.sum()), dtype=np.int64),
        ],
        grid,
        n_cells,
    )

    shape = (n_demes, n_states, grid.size)
    return Trajectories(
        time=grid,
        infectious=infectious.reshape(shape),
        sampled=sampled.reshape(shape),
        incidence=incidence.reshape(shape),
    )


def _state_lookup(layout: GenomeLayout) -> np.ndarray:
    """Таблица «индекс гаплотипа -> течение» для векторной адресации."""
    table = np.zeros(layout.haplotypes_number, dtype=np.int64)
    for state in (HostState.ACUTE, HostState.CHRONIC):
        table[layout.block(state)] = int(state)
    # недостижимые маркеры 'C'/'G' формально относим к обычному течению
    return table


def _accumulate(times, cells, deltas, grid: np.ndarray, n_cells: int) -> np.ndarray:
    """Кумулятивные суммы по ячейкам, снятые на сетке ``grid``."""
    result = np.zeros((n_cells, grid.size))
    if not times:
        return result
    all_times = np.concatenate(times)
    all_cells = np.concatenate(cells)
    all_deltas = np.concatenate(deltas)
    if all_times.size == 0:
        return result
    order = np.argsort(all_times, kind="stable")
    all_times, all_cells, all_deltas = (
        all_times[order],
        all_cells[order],
        all_deltas[order],
    )
    for cell_index in range(n_cells):
        mask = all_cells == cell_index
        if not mask.any():
            continue
        running = np.cumsum(all_deltas[mask])
        positions = np.searchsorted(all_times[mask], grid, side="right")
        padded = np.concatenate(([0], running))
        result[cell_index] = padded[positions]
    return result


#  Сводные показатели эпидемии


@dataclass(frozen=True)
class EpidemicSummary:
    """Показатели, по которым сравниваются однородная и стратифицированная модели."""

    label: str
    duration: float
    peak_time: float
    peak_prevalence: float
    total_infections: float
    total_samples: float
    decay_to_10: float
    decay_to_1: float
    decay_to_01: float
    tail_mass: float
    late_decay_rate: float
    chronic_peak: float
    chronic_share_late: float

    def as_row(self) -> dict[str, float | str]:
        return {
            "модель": self.label,
            "длительность, сут": round(self.duration, 1),
            "пик, сут": round(self.peak_time, 1),
            "высота пика": round(self.peak_prevalence),
            "всего заражений": round(self.total_infections),
            "секвенировано": round(self.total_samples),
            "пик -> 10 %, сут": round(self.decay_to_10, 1),
            "пик -> 1 %, сут": round(self.decay_to_1, 1),
            "пик -> 0,1 %, сут": round(self.decay_to_01, 1),
            "доля чел.-дней в хвосте": round(self.tail_mass, 4),
            "темп спада хвоста, 1/сут": round(self.late_decay_rate, 4),
            "пик затяжных": round(self.chronic_peak),
            "доля затяжных в хвосте": round(self.chronic_share_late, 3),
        }


def _integrate(values: np.ndarray, time: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None) or np.trapz
    return float(integrator(values, time))


def _first_crossing(total: np.ndarray, start: int, level: float) -> int | None:
    """Первый после ``start`` индекс, где кривая опустилась ниже ``level``."""
    below = np.nonzero(total[start:] < level)[0]
    return int(start + below[0]) if below.size else None


def summarise(
    trajectories: Trajectories, label: str, *, tail_threshold: float = 0.1
) -> EpidemicSummary:
    """Свести траекторию к набору чисел, включая метрики «длинного хвоста».

    Метрики спада устроены так, чтобы не зависеть от высоты пика: волна
    нормируется на собственный пик, и измеряется время падения до 10 %, 1 % и
    0,1 % от него. «Темп спада хвоста» — показатель экспоненты, подогнанной на
    участке между уровнями 1 % и 0,1 %. Его содержательный смысл: в однородной
    модели он равен скорости выбытия обычного течения, а в стратифицированной
    выходит на гораздо меньшую скорость выбытия затяжного — именно это и есть
    «длинный хвост», а не просто более высокая кривая.
    """
    time = trajectories.time
    total = trajectories.total
    if total.size == 0 or total.max() <= 0:
        return EpidemicSummary(label, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    peak_index = int(np.argmax(total))
    peak_value = float(total[peak_index])

    index_10 = _first_crossing(total, peak_index, tail_threshold * peak_value)
    index_1 = _first_crossing(total, peak_index, 0.01 * peak_value)
    index_01 = _first_crossing(total, peak_index, 0.001 * peak_value)

    def elapsed(index):
        return float(time[index] - time[peak_index]) if index is not None else float("nan")

    person_days = _integrate(total, time)
    tail_mass = 0.0
    if index_10 is not None and person_days > 0:
        tail_mass = _integrate(total[index_10:], time[index_10:]) / person_days

    late_decay_rate = _fit_decay(time, total, index_1, index_01)
    if np.isnan(late_decay_rate):  # хвост не дотянул до 0,1 % — берём предыдущее окно
        late_decay_rate = _fit_decay(time, total, index_10, index_1)

    reference = index_10 if index_10 is not None else peak_index
    return EpidemicSummary(
        label=label,
        duration=float(time[-1]),
        peak_time=float(time[peak_index]),
        peak_prevalence=peak_value,
        total_infections=float(trajectories.incidence.sum(axis=(0, 1))[-1]),
        total_samples=float(trajectories.sampled.sum(axis=(0, 1))[-1]),
        decay_to_10=elapsed(index_10),
        decay_to_1=elapsed(index_1),
        decay_to_01=elapsed(index_01),
        tail_mass=float(tail_mass),
        late_decay_rate=late_decay_rate,
        chronic_peak=float(trajectories.chronic_load.max()),
        chronic_share_late=float(trajectories.chronic_share[reference]),
    )


def _fit_decay(time: np.ndarray, total: np.ndarray, start: int | None, stop: int | None) -> float:
    """Показатель экспоненты, подогнанной по логарифму кривой на [start, stop]."""
    if start is None or stop is None or stop <= start + 2:
        return float("nan")
    window = slice(start, stop + 1)
    positive = total[window] > 0
    if positive.sum() <= 2:
        return float("nan")
    slope = np.polyfit(time[window][positive], np.log(total[window][positive]), 1)[0]
    return float(-slope)


#  Диагностика кодировки


def inheritance_diagnostics(chain: EventChain, layout: GenomeLayout) -> dict[str, float]:
    """Измерить, насколько велик дефект наследования служебного маркера.

    Маркер передаётся вместе с гаплотипом, поэтому заражённый от носителя
    затяжного течения тоже стартует затяжным, хотя в эталонной модели всякое
    заражение должно начинаться с обычного течения. Доля таких заражений —
    прямая эмпирическая мера смещения; аналитическая оценка сверху для неё
    даётся методом ``LinearisedEpidemic.inheritance_overhead``.
    """
    state_of = _state_lookup(layout)
    is_new_infection = (chain.kind == BIRTH) | (chain.kind == MIGRATION)
    donors = chain.haplotype[is_new_infection]
    total = int(donors.size)
    if total == 0:
        return {
            "new_infections": 0.0,
            "inherited_chronic": 0.0,
            "inherited_chronic_share": 0.0,
            "progressions": 0.0,
        }
    inherited = int((state_of[donors] == int(HostState.CHRONIC)).sum())

    is_mutation = chain.kind == MUTATION
    from_state = state_of[chain.haplotype[is_mutation]]
    to_state = state_of[chain.new_haplotype[is_mutation]]
    progressions = int(
        ((from_state == int(HostState.ACUTE)) & (to_state == int(HostState.CHRONIC))).sum()
    )
    return {
        "new_infections": float(total),
        "inherited_chronic": float(inherited),
        "inherited_chronic_share": inherited / total,
        "progressions": float(progressions),
    }


def bias_report(chain: EventChain, layout: GenomeLayout, module, trajectories: Trajectories):
    """Оценить смещение кодировки в человеко-днях инфекции.

    Каждое заражение, «ошибочно» стартовавшее затяжным, проводит в инфекционном
    состоянии в среднем ``mean_duration_chronic`` суток вместо
    ``mean_infectious_period``. Отношение суммарного избытка к полному числу
    человеко-дней и есть та ошибка, которую вносит отказ от отдельного
    компартмента.
    """
    diagnostics = inheritance_diagnostics(chain, layout)
    excess_per_case = module.chronic.mean_duration - module.mean_infectious_period
    excess_days = diagnostics["inherited_chronic"] * max(excess_per_case, 0.0)
    person_days = _integrate(trajectories.total, trajectories.time)
    diagnostics["excess_person_days"] = float(excess_days)
    diagnostics["person_days"] = float(person_days)
    diagnostics["excess_person_days_share"] = (
        float(excess_days / person_days) if person_days > 0 else 0.0
    )
    return diagnostics


#  Выборка и генеалогия


def sample_table(simulator, layout: GenomeLayout) -> dict[str, np.ndarray]:
    """Таблица секвенированных случаев: время, дем, гаплотип, течение."""
    time, population, haplotype = simulator.output_sample_data(output_print=True)
    time = np.asarray(time, dtype=float)
    population = np.asarray(population, dtype=np.int64)
    haplotype = np.asarray(haplotype, dtype=np.int64)
    state_of = _state_lookup(layout)
    return {
        "time": time,
        "deme": population,
        "haplotype": haplotype,
        "state": state_of[haplotype] if haplotype.size else haplotype,
        "genome": haplotype % layout.block_size if haplotype.size else haplotype,
    }


def tip_annotations(simulator, layout: GenomeLayout) -> dict[str, np.ndarray]:
    """Сопоставить листья генеалогии с метаданными выборки.

    Идентификаторы узлов в VGsim раздаются при обратном проходе по цепочке
    событий, поэтому листья не занимают непрерывный диапазон. Надёжный ключ —
    время: времена событий попарно различны, и время листа в точности равно
    времени соответствующего события SAMPLING.
    """
    tree, times, _mutations, _populations = simulator.simulation.output_tree_mutations()
    node_times = np.asarray(times, dtype=float)
    samples = sample_table(simulator, layout)
    index = {float(t): i for i, t in enumerate(samples["time"])}

    node_ids, matched = [], []
    for node_id, node_time in enumerate(node_times):
        position = index.get(float(node_time))
        if position is not None:
            node_ids.append(node_id)
            matched.append(position)
    node_ids_arr = np.asarray(node_ids, dtype=np.int64)
    matched_arr = np.asarray(matched, dtype=np.int64)
    return {
        "node": node_ids_arr,
        "time": samples["time"][matched_arr],
        "deme": samples["deme"][matched_arr],
        "state": samples["state"][matched_arr],
        "genome": samples["genome"][matched_arr],
    }


def terminal_branch_lengths(simulator, layout: GenomeLayout) -> dict[str, np.ndarray]:
    """Длины терминальных ветвей листьев с разбивкой по течению инфекции.

    Затяжные носители дольше остаются инфекционными, поэтому их линии реже
    коалесцируют перед взятием пробы: длинные терминальные ветви — прямое
    филогенетическое проявление субпопуляции хронических пациентов.
    """
    tree, times, _mutations, _populations = simulator.simulation.output_tree_mutations()
    parents = np.asarray(tree, dtype=np.int64)
    node_times = np.asarray(times, dtype=float)
    tips = tip_annotations(simulator, layout)

    lengths = []
    for node in tips["node"]:
        parent = int(parents[node])
        lengths.append(abs(float(node_times[node] - node_times[parent])) if parent >= 0 else np.nan)
    return {"state": tips["state"], "deme": tips["deme"], "length": np.asarray(lengths)}
