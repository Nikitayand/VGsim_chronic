"""Графики для главы 3: кривые эпидемии, состав выборки, длины ветвей.

Оформление рассчитано на печать: тонкие линии, приглушённая сетка, подписи
серий прямо у линий (не только цветом — цвет продублирован типом штриха, чтобы
рисунок читался в чёрно-белой печати и при дальтонизме), одна ось ординат
на график.
"""

from __future__ import annotations

import numpy as np

from .analysis import Trajectories, terminal_branch_lengths
from .encoding import HostState

__all__ = [
    "apply_style",
    "plot_prevalence",
    "plot_composition",
    "plot_normalised_tail",
    "plot_sampling",
    "plot_branch_lengths",
    "plot_trees_side_by_side",
]

#: Первые три слота проверенной категориальной палитры (все пары проходят
#: пороги различимости, в том числе при дальтонизме).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
DASHES = ("-", "--", ":")
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d3"


def apply_style() -> None:
    """Единое оформление для всех рисунков работы."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.0),
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _label_at(axis, x, y, index, text, color, offset=(4, 0), ha="left") -> None:
    """Прямая подпись у линии — идентичность передаётся не только цветом."""
    if index is None:
        return
    axis.annotate(
        text,
        xy=(x[index], y[index]),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
        ha=ha,
        clip_on=False,
    )


def _last_finite(y) -> int | None:
    finite = np.isfinite(y)
    return int(np.nonzero(finite)[0][-1]) if finite.any() else None


def _peak(y) -> int | None:
    return int(np.argmax(y)) if np.size(y) and np.isfinite(y).any() else None


def _crossing(y, level: float) -> int | None:
    above = np.nonzero(np.asarray(y) >= level)[0]
    return int(above[0]) if above.size else None


def _finish(figure, axis, path, title, xlabel, ylabel):
    axis.set_title(title, color=INK, loc="left", pad=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc="upper right")
    figure.savefig(path)
    return path


def plot_prevalence(runs, path: str, *, title: str = "Численность инфицированных"):
    """Кривые распространённости для нескольких моделей на одной оси."""
    import matplotlib.pyplot as plt

    apply_style()
    figure, axis = plt.subplots()
    for index, run in enumerate(runs):
        color, dash = SERIES[index % len(SERIES)], DASHES[index % len(DASHES)]
        time, total = run.trajectories.time, run.trajectories.total
        axis.plot(time, total, dash, color=color, label=run.label)
        _label_at(axis, time, total, _peak(total), run.label, color, offset=(6, 8 - 12 * index))
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    return _finish(figure, axis, path, title, "время, сут", "инфицированных, человек")


def plot_composition(run, path: str, *, title: str | None = None, log: bool = False):
    """Разложение стратифицированной модели на течения инфекции.

    :param log: логарифмическая ось ординат. На ней видно, что после спада
        волны почти вся оставшаяся распространённость — это затяжные носители.
    """
    import matplotlib.pyplot as plt

    apply_style()
    trajectories: Trajectories = run.trajectories
    series = [
        ("всего", trajectories.total, SERIES[0], DASHES[0]),
        ("обычное течение", trajectories.by_state(HostState.ACUTE), SERIES[1], DASHES[1]),
        ("затяжное течение", trajectories.by_state(HostState.CHRONIC), SERIES[2], DASHES[2]),
    ]
    figure, axis = plt.subplots()
    for index, (label, values, color, dash) in enumerate(series):
        plotted = np.where(values > 0, values, np.nan) if log else values
        axis.plot(trajectories.time, plotted, dash, color=color, label=label)
        anchor = _last_finite(plotted) if log else _peak(values)
        _label_at(
            axis,
            trajectories.time,
            plotted,
            anchor,
            label,
            color,
            offset=(6, 10 - 12 * index),
        )
    axis.set_xlim(left=0)
    if log:
        axis.set_yscale("log")
        axis.set_ylim(bottom=1)
    else:
        axis.set_ylim(bottom=0)
    return _finish(
        figure,
        axis,
        path,
        title or f"Состав инфицированных: {run.label}",
        "время, сут",
        "инфицированных, человек",
    )


def plot_normalised_tail(runs, path: str, *, threshold: float = 1e-3):
    """Кривые, нормированные на высоту пика и совмещённые по пику.

    Именно на этом рисунке виден «длинный хвост»: после совмещения пиков
    остаётся ровно то различие в скорости спада, которое вносит субпопуляция
    затяжного течения.
    """
    import matplotlib.pyplot as plt

    apply_style()
    figure, axis = plt.subplots()
    for index, run in enumerate(runs):
        time = run.trajectories.time
        total = run.trajectories.total
        if total.max() <= 0:
            continue
        peak = int(np.argmax(total))
        normalised = total / total[peak]
        shifted = time - time[peak]
        mask = (shifted >= 0) & (normalised > threshold)
        color, dash = SERIES[index % len(SERIES)], DASHES[index % len(DASHES)]
        axis.plot(shifted[mask], normalised[mask], dash, color=color, label=run.label)
        _label_at(
            axis,
            shifted[mask],
            normalised[mask],
            _last_finite(normalised[mask]),
            run.label,
            color,
            offset=(6, -6),
        )
    axis.set_yscale("log")
    axis.set_xlim(left=0)
    return _finish(
        figure,
        axis,
        path,
        "Спад волны после пика (нормировка на высоту пика)",
        "время от пика, сут",
        "доля от пиковой численности",
    )


def plot_sampling(run, path: str):
    """Накопленное число секвенированных случаев по течениям инфекции."""
    import matplotlib.pyplot as plt

    apply_style()
    trajectories: Trajectories = run.trajectories
    figure, axis = plt.subplots()
    for index, state in enumerate((HostState.ACUTE, HostState.CHRONIC)):
        values = trajectories.sampled[:, int(state), :].sum(axis=0)
        color, dash = SERIES[index], DASHES[index]
        label = state.title
        axis.plot(trajectories.time, values, dash, color=color, label=label)
        _label_at(axis, trajectories.time, values, _last_finite(values), label, color)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    return _finish(
        figure,
        axis,
        path,
        f"Накопленная выборка: {run.label}",
        "время, сут",
        "секвенировано, случаев",
    )


def plot_branch_lengths(run, path: str, *, bins: int = 40):
    """Эмпирические функции распределения длин терминальных ветвей.

    Сравниваются листья, взятые от носителей обычного и затяжного течения:
    у затяжных линий коалесценция откладывается, и ветви систематически длиннее.
    """
    import matplotlib.pyplot as plt

    apply_style()
    data = terminal_branch_lengths(run.simulator, run.layout)
    figure, axis = plt.subplots()
    drawn = False
    for index, state in enumerate((HostState.ACUTE, HostState.CHRONIC)):
        values = data["length"][data["state"] == int(state)]
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        drawn = True
        ordered = np.sort(values)
        ecdf = np.arange(1, ordered.size + 1) / ordered.size
        color, dash = SERIES[index], DASHES[index]
        label = f"{state.title} (n={ordered.size})"
        axis.plot(ordered, ecdf, dash, color=color, label=label)
        median = _crossing(ecdf, 0.5)
        _label_at(axis, ordered, ecdf, median, state.title, color, offset=(6, -10))
    if not drawn:
        axis.text(0.5, 0.5, "недостаточно листьев", ha="center", transform=axis.transAxes)
    axis.set_ylim(0, 1.02)
    axis.set_xlim(left=0)
    return _finish(
        figure,
        axis,
        path,
        f"Длины терминальных ветвей: {run.label}",
        "длина терминальной ветви, сут",
        "доля листьев не длиннее",
    )


#  Генеалогии


def _tree_layout(tree):
    """Координаты узлов: x — время от корня, y — порядок листьев после лесенки."""
    import numpy as np

    order = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        order.append(node)
        kids = sorted(tree.children[node], key=lambda c: -len(tree.children[c]))
        stack.extend(kids)

    x = tree.depths()
    y = np.zeros(tree.size)
    counter = 0
    for node in reversed(order):
        if not tree.children[node]:
            y[node] = counter
            counter += 1
    for node in reversed(order):
        kids = tree.children[node]
        if kids:
            y[node] = float(np.mean([y[k] for k in kids]))
    return x, y


def plot_trees_side_by_side(tree_a, tree_b, courses, path, labels=None):
    """Два дерева рядом; листья второго раскрашены по течению инфекции.

    :param courses: отображение «метка листа -> течение» (из файла разметки).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    labels = labels or ("Старая модель: без хронических пациентов",
                        "Новая модель: с субпопуляцией затяжного течения")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 5.6), sharey=False)
    palette = {"acute": SERIES[0], "chronic": SERIES[1]}
    # общая шкала времени: иначе разница в высоте деревьев не видна глазом
    span = max(float(tree_a.depths().max()), float(tree_b.depths().max()))

    for axis, tree, title, colour_tips in zip(
        axes, (tree_a, tree_b), labels, (False, True)
    ):
        x, y = _tree_layout(tree)
        segments_h, segments_v, colours_h = [], [], []
        for node in range(tree.size):
            up = tree.parent[node]
            if up >= 0:
                segments_h.append([(x[up], y[node]), (x[node], y[node])])
                course = courses.get(tree.label[node]) if colour_tips else None
                if course and not tree.children[node]:
                    colours_h.append(palette.get(course, INK_SOFT))
                else:
                    colours_h.append(INK_SOFT)
            kids = tree.children[node]
            if kids:
                segments_v.append(
                    [(x[node], min(y[k] for k in kids)), (x[node], max(y[k] for k in kids))]
                )
        from matplotlib.collections import LineCollection

        axis.add_collection(
            LineCollection(segments_h, colors=colours_h, linewidths=0.7)
        )
        axis.add_collection(
            LineCollection(segments_v, colors=INK_SOFT, linewidths=0.5)
        )
        axis.set_xlim(-0.02 * span, 1.04 * span)
        axis.set_ylim(-1, y.max() + 1)
        axis.set_yticks([])
        axis.set_xlabel("время от корня, сут")
        axis.set_title(title, loc="left", fontsize=9, pad=8)
        axis.grid(axis="y", visible=False)

    if any(courses.values()):
        from matplotlib.lines import Line2D

        axes[1].legend(
            handles=[
                Line2D([0], [0], color=palette["acute"], lw=1.6, label="обычное течение"),
                Line2D([0], [0], color=palette["chronic"], lw=1.6, label="затяжное течение"),
            ],
            loc="lower right",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(path)
    return path
