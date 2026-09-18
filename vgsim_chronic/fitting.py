"""Подгонка моделей к реальным рядам и метрики согласия.

Задача главы 3.3 поставлена узко и симметрично. Ни однородная, ни
стратифицированная модель не содержат ни вакцинации, ни немедикаментозных мер,
ни смены вариантов, поэтому речь не идёт о механистической реконструкции
конкретной волны. Обе модели подгоняются к одному и тому же ряду **тремя
одинаковыми свободными параметрами**:

* ``R0`` — из общей сетки значений;
* ``shift`` — положение модельного пика относительно наблюдаемого, сут;
* ``scale`` — множитель «инфицированных на одну занятую койку».

Параметры самого модуля хронических пациентов (доля затяжных случаев,
длительность затяжного течения, относительная заразность) взяты из литературы
и **не подгоняются**. Поэтому улучшение согласия нельзя объяснить большей
свободой подгонки: свободных параметров у моделей поровну.

Подгонка ведётся обычным методом наименьших квадратов по числу коек — то есть
ровно так, как её выполнил бы эпидемиолог, минимизирующий RMSE. Все остальные
метрики считаются уже на этой подгонке: вопрос не «какая модель лучше при
своей любимой метрике», а «где именно накапливается расхождение при
стандартной подгонке».
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from .analysis import collect_trajectories
from .encoding import GenomeLayout
from .model import ChronicModel
from .params import ModelConfig
from .theory import calibrate_transmission

__all__ = [
    "ObservedWave",
    "ModelCurve",
    "FitResult",
    "load_wave",
    "simulate_curve",
    "build_curve_bank",
    "save_curve_bank",
    "load_curve_bank",
    "fit_wave",
    "goodness_of_fit",
]

#: Порог хвоста: наблюдаемая занятость упала ниже этой доли пика.
TAIL_FRACTION = 0.20


#  Наблюдаемый ряд


@dataclass(frozen=True)
class ObservedWave:
    """Одна волна реальной статистики занятости коек."""

    name: str
    dates: list[str]
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.values.size)

    @property
    def peak_index(self) -> int:
        return int(np.argmax(self.values))

    @property
    def peak_value(self) -> float:
        return float(self.values.max())

    @property
    def tail_mask(self) -> np.ndarray:
        """После пика и ниже ``TAIL_FRACTION`` от него — область «длинного хвоста».

        Границу задаёт сам ряд, а не модель, поэтому обе модели оцениваются на
        одном и том же множестве точек.
        """
        mask = np.zeros(self.size, dtype=bool)
        mask[self.peak_index :] = True
        return mask & (self.values <= TAIL_FRACTION * self.peak_value)

    def observed_tail_decay(self, window: int = 60) -> float:
        """Наблюдаемый темп экспоненциального спада хвоста, 1/сут."""
        tail = np.nonzero(self.tail_mask)[0][:window]
        if tail.size < 5:
            return float("nan")
        values = self.values[tail]
        positive = values > 0
        if positive.sum() < 5:
            return float("nan")
        slope = np.polyfit(tail[positive], np.log(values[positive]), 1)[0]
        return float(-slope)


def load_wave(path: str, name: str | None = None) -> ObservedWave:
    """Прочитать CSV с колонками ``date`` и ``occupancy``."""
    dates: list[str] = []
    values: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dates.append(row["date"])
            values.append(float(row["occupancy"]))
    label = name or os.path.splitext(os.path.basename(path))[0]
    return ObservedWave(name=label, dates=dates, values=np.asarray(values, dtype=float))


#  Банк модельных кривых


@dataclass(frozen=True)
class ModelCurve:
    """Форма модельной волны: распространённость как функция времени от пика.

    Кривая нормирована на собственный пик, поэтому масштаб при подгонке
    остаётся свободным параметром, а стохастический разброс момента взлёта
    снимается совмещением повторов по их собственным пикам.

    :param shape: значение второго параметра формы — средний инфекционный
        период у однородной модели, доля затяжных случаев у стратифицированной.
    """

    label: str
    r0: float
    rel_time: np.ndarray
    value: np.ndarray
    shape: float = 0.0

    def sample(self, offsets: np.ndarray) -> np.ndarray:
        """Значения кривой в точках ``offsets`` (сутки от пика), вне носителя — 0."""
        return np.interp(offsets, self.rel_time, self.value, left=0.0, right=0.0)


def simulate_curve(
    config: ModelConfig,
    r0: float,
    *,
    shape: float = 0.0,
    seeds: tuple[int, ...] = (11, 22, 33),
    epidemic_time: float = 600.0,
    iterations: int = 40_000_000,
    steps: int = 1200,
    horizon: tuple[float, float] = (-90.0, 300.0),
    grid_step: float = 1.0,
) -> ModelCurve:
    """Средняя по повторам форма волны для заданного ``R0``."""
    tuned = calibrate_transmission(config, r0)
    grid = np.arange(horizon[0], horizon[1] + grid_step, grid_step)
    stack = []
    for seed in seeds:
        model = ChronicModel(tuned)
        simulator = model.build(seed=seed)
        simulator.simulate(iterations, epidemic_time=epidemic_time)
        trajectories = collect_trajectories(simulator, model.layout, steps=steps)
        total = trajectories.total
        if total.max() <= 0:
            continue
        peak = int(np.argmax(total))
        shifted = trajectories.time - trajectories.time[peak]
        stack.append(np.interp(grid, shifted, total / total[peak], left=0.0, right=0.0))
    if not stack:
        raise RuntimeError(f"при R0={r0} эпидемия не состоялась ни в одном повторе")
    mean = np.mean(np.vstack(stack), axis=0)
    return ModelCurve(
        label=tuned.label, r0=float(r0), rel_time=grid, value=mean, shape=float(shape)
    )


def build_curve_bank(variants, r0_grid, *, progress=None, **kwargs) -> list[ModelCurve]:
    """Двумерный банк форм: сетка ``R0`` на сетку второго параметра формы.

    :param variants: последовательность пар ``(значение параметра формы, конфигурация)``.
    """
    bank: list[ModelCurve] = []
    for shape, config in variants:
        for r0 in r0_grid:
            if progress is not None:
                progress(shape, float(r0))
            bank.append(simulate_curve(config, float(r0), shape=float(shape), **kwargs))
    return bank


def save_curve_bank(path: str, bank: list[ModelCurve]) -> None:
    np.savez_compressed(
        path,
        label=np.array([c.label for c in bank]),
        r0=np.array([c.r0 for c in bank]),
        shape=np.array([c.shape for c in bank]),
        rel_time=np.vstack([c.rel_time for c in bank]),
        value=np.vstack([c.value for c in bank]),
    )


def load_curve_bank(path: str) -> list[ModelCurve]:
    data = np.load(path, allow_pickle=False)
    shapes = data["shape"] if "shape" in data.files else np.zeros(len(data["r0"]))
    return [
        ModelCurve(
            label=str(data["label"][i]),
            r0=float(data["r0"][i]),
            rel_time=data["rel_time"][i],
            value=data["value"][i],
            shape=float(shapes[i]),
        )
        for i in range(len(data["r0"]))
    ]


#  Безразмерное число формы


def shape_statistics(
    time: np.ndarray, values: np.ndarray, *, window: int = 60
) -> dict[str, float]:
    """Ширина волны на половине пика, темп спада хвоста и их произведение.

    Произведение ``П = W50 · lambda`` безразмерно, не зависит ни от масштаба по
    вертикали, ни от сдвига по времени — то есть от обоих «технических»
    параметров подгонки. Для однородной модели оно не зависит **и** от среднего
    инфекционного периода: изменение единственного временного масштаба при
    фиксированном R0 — это просто перемасштабирование времени, при котором
    ширина растёт во столько же раз, во сколько падает темп спада. Поэтому у
    однородной модели П — функция одного R0, и достижимые значения образуют
    кривую, а не область.

    У стратифицированной модели появляется вторая ручка — вес затяжной
    компоненты, — и он меняет отношение «узость пика к медленности хвоста»
    независимо от R0. Это и делает П пригодным для проверки: наблюдаемое
    значение либо попадает в достижимый диапазон семейства, либо нет.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    peak = int(np.argmax(values))
    half = 0.5 * values[peak]

    left = np.nonzero(values[: peak + 1] >= half)[0]
    right = np.nonzero(values[peak:] >= half)[0]
    width = (
        float(time[peak + right[-1]] - time[left[0]])
        if left.size and right.size
        else float("nan")
    )

    index = np.nonzero(
        (np.arange(values.size) >= peak)
        & (values <= TAIL_FRACTION * values[peak])
        & (values > 0)
    )[0][:window]
    decay = float("nan")
    if index.size >= 5:
        decay = float(-np.polyfit(time[index], np.log(values[index]), 1)[0])

    return {"width_half": width, "tail_decay": decay, "shape_number": width * decay}


#  Метрики согласия


def goodness_of_fit(
    observed: np.ndarray,
    predicted: np.ndarray,
    tail_mask: np.ndarray,
    n_params: int,
) -> dict[str, float]:
    """Четыре метрики согласия на одной и той же подгонке.

    * ``rmse`` — стандартная среднеквадратичная ошибка в койках. Определяется
      в основном окрестностью пика: там значения на порядок больше, и вклад
      хвоста в сумму квадратов почти не виден.
    * ``rmse_tail`` — та же ошибка, но только на хвосте волны. Именно здесь
      модели различаются механистически.
    * ``r2`` — доля объяснённой дисперсии; безразмерна, поэтому волны разного
      размера сопоставимы между собой.
    * ``aic`` / ``bic`` — информационные критерии для гауссовой ошибки,
      ``n·ln(RSS/n) + 2k`` и ``n·ln(RSS/n) + k·ln n``. Они штрафуют модель за
      число параметров и отвечают на вопрос, окупается ли усложнение.
    * ``rmsle`` — среднеквадратичная ошибка логарифма; приведена как проверка
      устойчивости вывода к выбору шкалы.
    """
    residual = observed - predicted
    rss = float(np.sum(residual**2))
    size = int(observed.size)
    rmse = float(np.sqrt(rss / size))

    tail_rmse = float("nan")
    if tail_mask.any():
        tail_rmse = float(np.sqrt(np.mean(residual[tail_mask] ** 2)))

    variance = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - rss / variance) if variance > 0 else float("nan")

    log_error = np.log1p(np.maximum(observed, 0)) - np.log1p(np.maximum(predicted, 0))
    rmsle = float(np.sqrt(np.mean(log_error**2)))

    aic = float(size * np.log(rss / size) + 2 * n_params) if rss > 0 else float("-inf")
    bic = (
        float(size * np.log(rss / size) + n_params * np.log(size))
        if rss > 0
        else float("-inf")
    )
    return {
        "rmse": rmse,
        "rmse_tail": tail_rmse,
        "r2": r2,
        "rmsle": rmsle,
        "aic": aic,
        "bic": bic,
        "rss": rss,
        "n": float(size),
        "n_params": float(n_params),
    }


#  Подгонка


@dataclass
class FitResult:
    """Результат подгонки одной модели к одной волне."""

    model: str
    wave: str
    r0: float
    shift: float
    scale: float
    predicted: np.ndarray
    shape: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def fitted_tail_decay(self) -> float:
        """Темп спада подогнанной кривой на том же окне, что и у данных."""
        values = self.predicted
        peak = int(np.argmax(values))
        after = values[peak:]
        low = np.nonzero(after <= TAIL_FRACTION * values[peak])[0]
        if low.size < 5:
            return float("nan")
        index = peak + low[:60]
        positive = values[index] > 0
        if positive.sum() < 5:
            return float("nan")
        slope = np.polyfit(index[positive], np.log(values[index][positive]), 1)[0]
        return float(-slope)


def fit_wave(
    wave: ObservedWave,
    bank: list[ModelCurve],
    *,
    model_name: str,
    n_params: int,
    shift_range: int = 45,
) -> FitResult:
    """Подобрать ``R0``, сдвиг и масштаб методом наименьших квадратов.

    Масштаб при фиксированных ``R0`` и сдвиге находится в замкнутой форме
    (регрессия через начало координат), поэтому перебираются только два
    параметра, и оптимум по ним находится точно на выбранной сетке.
    """
    index = np.arange(wave.size, dtype=float)
    best: FitResult | None = None
    best_rss = np.inf
    for curve in bank:
        for shift in range(-shift_range, shift_range + 1):
            offsets = index - (wave.peak_index + shift)
            shape = curve.sample(offsets)
            denominator = float(np.sum(shape**2))
            if denominator <= 0:
                continue
            scale = float(np.sum(wave.values * shape) / denominator)
            if scale <= 0:
                continue
            predicted = scale * shape
            rss = float(np.sum((wave.values - predicted) ** 2))
            if rss < best_rss:
                best_rss = rss
                best = FitResult(
                    model=model_name,
                    wave=wave.name,
                    r0=curve.r0,
                    shift=float(shift),
                    scale=scale,
                    predicted=predicted,
                    shape=curve.shape,
                )
    if best is None:
        raise RuntimeError(f"не удалось подогнать модель к волне {wave.name}")
    best.metrics = goodness_of_fit(wave.values, best.predicted, wave.tail_mask, n_params)
    return best
