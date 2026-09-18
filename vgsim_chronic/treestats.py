"""Сравнение генеалогий двух моделей: форма, ветви и расстояния между деревьями.

Деревья двух независимых прогонов **не имеют общих листьев**, поэтому обычные
метрики расстояния между филогениями (Робинсона — Фулдса, Кендалла — Колейн)
неприменимы: они определены только для деревьев на одном наборе таксонов.
Здесь сравнение устроено иначе — через величины, определённые для дерева самого
по себе:

* статистики формы (несбалансированность, локальные конфигурации),
* распределения длин ветвей,
* кривая числа линий во времени,
* спектральная плотность модифицированного лапласиана дерева.

Последняя — метод Левитуса и Морлона: собственные значения матрицы попарных
филогенетических расстояний сравниваются как распределения, что и даёт
расстояние между деревьями с разными наборами листьев.

Чтобы статистики формы были сопоставимы, оба дерева прореживаются до
одинакового числа листьев, и прореживание повторяется многократно: результат —
распределение каждой статистики, а не одно число.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Tree",
    "parse_newick",
    "read_newick",
    "shape_statistics",
    "branch_statistics",
    "ltt_curve",
    "spectral_density",
    "compare_trees",
    "ks_distance",
]

_TOKEN = re.compile(r"\s*([(),;])\s*|\s*([^(),;:]+)?\s*(?::\s*(-?[0-9.eE+-]+))?\s*")


@dataclass
class Tree:
    """Укоренённое дерево в виде массивов.

    :param parent: индекс родителя, -1 у корня.
    :param length: длина ветви к родителю.
    :param label: метка узла (у VGsim это номер узла генеалогии).
    :param children: списки потомков.
    """

    parent: np.ndarray
    length: np.ndarray
    label: list[str]
    children: list[list[int]] = field(default_factory=list)

    # базовое

    @property
    def size(self) -> int:
        return int(self.parent.size)

    @property
    def root(self) -> int:
        return int(np.nonzero(self.parent < 0)[0][0])

    @property
    def is_tip(self) -> np.ndarray:
        return np.array([len(c) == 0 for c in self.children], dtype=bool)

    @property
    def tips(self) -> np.ndarray:
        return np.nonzero(self.is_tip)[0]

    @property
    def n_tips(self) -> int:
        return int(self.is_tip.sum())

    def depths(self) -> np.ndarray:
        """Расстояние от корня до каждого узла вдоль ветвей."""
        depth = np.zeros(self.size)
        for node in self.preorder():
            up = self.parent[node]
            if up >= 0:
                depth[node] = depth[up] + self.length[node]
        return depth

    def node_depths(self) -> np.ndarray:
        """Число рёбер от корня до узла."""
        steps = np.zeros(self.size, dtype=np.int64)
        for node in self.preorder():
            up = self.parent[node]
            if up >= 0:
                steps[node] = steps[up] + 1
        return steps

    def preorder(self) -> list[int]:
        order, stack = [], [self.root]
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(self.children[node])
        return order

    def clade_sizes(self) -> np.ndarray:
        """Число листьев под каждым узлом."""
        sizes = np.zeros(self.size, dtype=np.int64)
        for node in reversed(self.preorder()):
            if not self.children[node]:
                sizes[node] = 1
            else:
                sizes[node] = sum(sizes[c] for c in self.children[node])
        return sizes

    # операции

    def subsample(self, n_tips: int, rng: np.random.Generator) -> "Tree":
        """Оставить ``n_tips`` случайных листьев, схлопнув проходные узлы.

        Длины ветвей суммируются, поэтому времена узлов сохраняются.
        """
        tips = self.tips
        if n_tips >= tips.size:
            return self
        keep = set(int(t) for t in rng.choice(tips, size=n_tips, replace=False))
        alive = set(keep)
        for node in reversed(self.preorder()):
            if any(c in alive for c in self.children[node]):
                alive.add(node)

        new_index: dict[int, int] = {}
        parents: list[int] = []
        lengths: list[float] = []
        labels: list[str] = []
        children: list[list[int]] = []

        def add(node: int, parent_new: int, extra: float) -> None:
            kept = [c for c in self.children[node] if c in alive]
            # проходной узел: один живой потомок и сам не лист -> схлопываем.
            # В extra уходит длина ветви СХЛОПЫВАЕМОГО узла; длина ветви узла,
            # на котором цикл остановится, добавляется уже после него.
            while len(kept) == 1 and node not in keep:
                extra += self.length[node]
                node = kept[0]
                kept = [c for c in self.children[node] if c in alive]
            index = len(parents)
            new_index[node] = index
            parents.append(parent_new)
            lengths.append(float(self.length[node] + extra) if parent_new >= 0 else 0.0)
            labels.append(self.label[node])
            children.append([])
            if parent_new >= 0:
                children[parent_new].append(index)
            for child in kept:
                add(child, index, 0.0)

        root = self.root
        while True:
            kept = [c for c in self.children[root] if c in alive]
            if len(kept) != 1:
                break
            root = kept[0]
        add(root, -1, 0.0)
        return Tree(
            parent=np.asarray(parents, dtype=np.int64),
            length=np.asarray(lengths, dtype=float),
            label=labels,
            children=children,
        )


#  Разбор newick


def parse_newick(text: str) -> Tree:
    """Разобрать строку newick в :class:`Tree` (итеративно, без рекурсии)."""
    text = text.strip()
    if not text.endswith(";"):
        text += ";"
    parents: list[int] = []
    lengths: list[float] = []
    labels: list[str] = []
    children: list[list[int]] = []
    stack: list[int] = []
    position = 0
    pending_parent = -1

    def new_node(parent: int) -> int:
        index = len(parents)
        parents.append(parent)
        lengths.append(0.0)
        labels.append("")
        children.append([])
        if parent >= 0:
            children[parent].append(index)
        return index

    while position < len(text):
        char = text[position]
        if char == "(":
            node = new_node(pending_parent if not stack else stack[-1])
            stack.append(node)
            pending_parent = -1
            position += 1
        elif char in ",)":
            if char == ")":
                node = stack.pop()
                position += 1
                match = re.match(r"([^(),;:]*)(?::(-?[0-9.eE+-]+))?", text[position:])
                if match:
                    labels[node] = (match.group(1) or "").strip()
                    if match.group(2):
                        lengths[node] = float(match.group(2))
                    position += match.end()
            else:
                position += 1
        elif char == ";":
            break
        else:
            match = re.match(r"([^(),;:]+)(?::(-?[0-9.eE+-]+))?", text[position:])
            if not match:
                position += 1
                continue
            node = new_node(stack[-1] if stack else -1)
            labels[node] = match.group(1).strip()
            if match.group(2):
                lengths[node] = float(match.group(2))
            position += match.end()

    return Tree(
        parent=np.asarray(parents, dtype=np.int64),
        length=np.asarray(lengths, dtype=float),
        label=labels,
        children=children,
    )


def read_newick(path: str) -> Tree:
    with open(path, encoding="utf-8") as handle:
        return parse_newick(handle.read())


#  Статистики формы


def shape_statistics(tree: Tree) -> dict[str, float]:
    """Статистики топологии, нормированные на число листьев.

    * ``sackin`` — индекс Сакина: средняя глубина листа в рёбрах.
    * ``colless`` — индекс Коллесса: средний по внутренним узлам модуль
      разности числа листьев в двух поддеревьях.
    * ``cherries`` — доля внутренних узлов, оба потомка которых листья.
    * ``pitchforks`` — доля клад ровно из трёх листьев.
    * ``il_fraction`` — доля внутренних узлов ровно с одним потомком-листом;
      растёт, когда дерево «лестничное».
    * ``max_ladder`` — длина самой длинной такой лестницы, в долях от числа листьев.
    """
    n = tree.n_tips
    if n < 3:
        return {k: float("nan") for k in
                ("sackin", "colless", "cherries", "pitchforks", "il_fraction", "max_ladder")}
    sizes = tree.clade_sizes()
    steps = tree.node_depths()
    is_tip = tree.is_tip
    internal = np.nonzero(~is_tip)[0]

    sackin = float(steps[is_tip].sum() / n)

    colless_total, colless_count = 0.0, 0
    cherries, pitchforks, il_nodes = 0, 0, 0
    for node in internal:
        kids = tree.children[node]
        if len(kids) == 2:
            colless_total += abs(sizes[kids[0]] - sizes[kids[1]])
            colless_count += 1
        tip_kids = sum(1 for k in kids if is_tip[k])
        if len(kids) == 2 and tip_kids == 2:
            cherries += 1
        if tip_kids == 1:
            il_nodes += 1
        if sizes[node] == 3:
            pitchforks += 1

    # длина лестницы: цепочка внутренних узлов, у каждого ровно один потомок-лист
    ladder = np.zeros(tree.size, dtype=np.int64)
    for node in reversed(tree.preorder()):
        if is_tip[node]:
            continue
        kids = tree.children[node]
        if sum(1 for k in kids if is_tip[k]) == 1:
            inner = [k for k in kids if not is_tip[k]]
            ladder[node] = 1 + (max(ladder[k] for k in inner) if inner else 0)

    return {
        "sackin": sackin,
        "colless": float(colless_total / colless_count) if colless_count else float("nan"),
        "cherries": float(cherries / n),
        "pitchforks": float(pitchforks / n),
        "il_fraction": float(il_nodes / max(internal.size, 1)),
        "max_ladder": float(ladder.max() / n),
    }


def branch_statistics(tree: Tree) -> dict[str, float]:
    """Длины терминальных и внутренних ветвей и высота дерева."""
    is_tip = tree.is_tip
    root = tree.root
    terminal = tree.length[is_tip]
    inner_mask = (~is_tip) & (np.arange(tree.size) != root)
    internal = tree.length[inner_mask]
    depths = tree.depths()
    return {
        "terminal_median": float(np.median(terminal)) if terminal.size else float("nan"),
        "terminal_mean": float(terminal.mean()) if terminal.size else float("nan"),
        "terminal_p90": float(np.percentile(terminal, 90)) if terminal.size else float("nan"),
        "internal_median": float(np.median(internal)) if internal.size else float("nan"),
        "terminal_share": float(terminal.sum() / (terminal.sum() + internal.sum()))
        if terminal.size and internal.size
        else float("nan"),
        "height": float(depths.max()),
    }


def ltt_curve(tree: Tree, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Число линий во времени на сетке, нормированной на высоту дерева.

    Линия появляется при ветвлении и исчезает при взятии пробы (лист),
    поэтому кривая считается по обоим типам событий.
    """
    depths = tree.depths()
    height = depths.max()
    if height <= 0:
        return np.zeros(points), np.zeros(points)
    is_tip = tree.is_tip
    events = []
    for node in range(tree.size):
        if is_tip[node]:
            events.append((depths[node], -1))
        elif len(tree.children[node]) >= 2:
            events.append((depths[node], len(tree.children[node]) - 1))
    events.sort()
    times = np.array([e[0] for e in events])
    deltas = np.array([e[1] for e in events])
    counts = np.cumsum(deltas) + 1
    grid = np.linspace(0.0, height, points)
    index = np.searchsorted(times, grid, side="right")
    padded = np.concatenate(([1], counts))
    return grid / height, padded[index].astype(float)


#  Спектральная плотность дерева


def spectral_density(
    tree: Tree, grid: np.ndarray | None = None, bandwidth: float = 0.12
) -> tuple[np.ndarray, np.ndarray]:
    """Плотность собственных значений модифицированного лапласиана дерева.

    Модифицированный лапласиан строится по матрице попарных филогенетических
    расстояний между узлами: ``L = D_deg - P``, где ``P`` — матрица расстояний,
    ``D_deg`` — диагональ её строчных сумм. Спектр такого оператора не зависит
    от имён листьев, поэтому годится для сравнения деревьев с разными выборками
    (метод Левитуса и Морлона). Плотность строится по логарифмам собственных
    значений гауссовым ядром.
    """
    distances = _pairwise_distances(tree)
    laplacian = np.diag(distances.sum(axis=1)) - distances
    values = np.linalg.eigvalsh(laplacian)
    values = values[values > 1e-9]
    if values.size == 0:
        raise ValueError("вырожденный спектр")
    logs = np.log(values)
    if grid is None:
        grid = np.linspace(logs.min() - 1.0, logs.max() + 1.0, 512)
    density = np.exp(-0.5 * ((grid[:, None] - logs[None, :]) / bandwidth) ** 2).sum(axis=1)
    density /= np.trapezoid(density, grid) if hasattr(np, "trapezoid") else np.trapz(density, grid)
    return grid, density


def _pairwise_distances(tree: Tree) -> np.ndarray:
    """Матрица филогенетических расстояний между всеми узлами дерева."""
    size = tree.size
    depths = tree.depths()
    order = tree.preorder()
    # предки каждого узла как битовые пути не нужны: считаем через восходящие суммы
    distances = np.zeros((size, size))
    for node in order:
        up = tree.parent[node]
        if up < 0:
            continue
        distances[node] = distances[up] + tree.length[node]
        distances[:, node] = distances[node]
        distances[node, node] = 0.0
        distances[node, up] = tree.length[node]
        distances[up, node] = tree.length[node]
    np.fill_diagonal(distances, 0.0)
    del depths
    return distances


#  Расстояния и сравнение


def ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Статистика Колмогорова — Смирнова между двумя выборками."""
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == 0 or right.size == 0:
        return float("nan")
    grid = np.concatenate([left, right])
    cdf_left = np.searchsorted(left, grid, side="right") / left.size
    cdf_right = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(cdf_left - cdf_right)))


def jensen_shannon(density_a: np.ndarray, density_b: np.ndarray, grid: np.ndarray) -> float:
    """Расстояние Йенсена — Шеннона между двумя плотностями на общей сетке."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    a = density_a / integrate(density_a, grid)
    b = density_b / integrate(density_b, grid)
    m = 0.5 * (a + b)

    def kl(p, q):
        mask = p > 1e-12
        return float(integrate(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12)), grid[mask]))

    return float(np.sqrt(max(0.5 * kl(a, m) + 0.5 * kl(b, m), 0.0)))


def compare_trees(
    tree_a: Tree,
    tree_b: Tree,
    *,
    n_tips: int = 500,
    replicates: int = 40,
    seed: int = 0,
) -> dict:
    """Сравнить два дерева по прореженным до одного размера копиям.

    Возвращает словарь со средними и стандартными отклонениями каждой
    статистики для обоих деревьев, размером эффекта (разность в единицах
    объединённого стандартного отклонения) и расстояниями, не требующими
    общих листьев.
    """
    rng = np.random.default_rng(seed)
    keys_shape = ("sackin", "colless", "cherries", "pitchforks", "il_fraction", "max_ladder")
    keys_branch = ("terminal_median", "terminal_mean", "terminal_p90", "internal_median",
                   "terminal_share", "height")
    collected: dict[str, dict[str, list[float]]] = {
        "a": {k: [] for k in keys_shape + keys_branch},
        "b": {k: [] for k in keys_shape + keys_branch},
    }
    ltt_a, ltt_b = [], []
    terminal_a, terminal_b = [], []

    for _ in range(replicates):
        sub_a = tree_a.subsample(n_tips, rng)
        sub_b = tree_b.subsample(n_tips, rng)
        for tag, sub in (("a", sub_a), ("b", sub_b)):
            stats = shape_statistics(sub)
            stats.update(branch_statistics(sub))
            for key in collected[tag]:
                collected[tag][key].append(stats[key])
        ltt_a.append(ltt_curve(sub_a)[1])
        ltt_b.append(ltt_curve(sub_b)[1])
        terminal_a.append(sub_a.length[sub_a.is_tip])
        terminal_b.append(sub_b.length[sub_b.is_tip])

    summary = {}
    for key in collected["a"]:
        a = np.array(collected["a"][key], dtype=float)
        b = np.array(collected["b"][key], dtype=float)
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
        summary[key] = {
            "a_mean": float(a.mean()),
            "a_sd": float(a.std(ddof=1)),
            "b_mean": float(b.mean()),
            "b_sd": float(b.std(ddof=1)),
            "effect": float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan"),
        }

    grid_rel = ltt_curve(tree_a.subsample(n_tips, rng))[0]
    ltt_mean_a = np.mean(np.vstack(ltt_a), axis=0)
    ltt_mean_b = np.mean(np.vstack(ltt_b), axis=0)

    distances = {
        "ks_terminal": ks_distance(np.concatenate(terminal_a), np.concatenate(terminal_b)),
        "ltt_max_gap": float(
            np.max(np.abs(ltt_mean_a / ltt_mean_a.max() - ltt_mean_b / ltt_mean_b.max()))
        ),
    }
    return {
        "summary": summary,
        "distances": distances,
        "ltt": {"grid": grid_rel, "a": ltt_mean_a, "b": ltt_mean_b},
    }
