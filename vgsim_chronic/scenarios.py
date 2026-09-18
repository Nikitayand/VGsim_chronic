"""Сценарии сравнения: однородная модель против стратифицированной.

Сравнение имеет смысл только при **одинаковом** R0: иначе разница в кривых
объясняется разным R0, а не наличием субпопуляции хронических пациентов.
Поэтому обе конфигурации прогоняются через :func:`calibrate_transmission`,
а базовая модель строится из стратифицированной по правилу «то же население,
тот же средний инфекционный период, та же доля секвенируемых».
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .analysis import (
    EpidemicSummary,
    bias_report,
    EventChain,
    Trajectories,
    collect_trajectories,
    inheritance_diagnostics,
    load_event_chain,
    sample_table,
    summarise,
    terminal_branch_lengths,
    tip_annotations,
)
from .encoding import GenomeLayout, HostState
from .model import ChronicModel
from .params import ModelConfig, homogeneous_config
from .theory import calibrate_transmission, linearise

__all__ = ["RunResult", "run_model", "compare", "ComparisonResult"]


@dataclass
class RunResult:
    """Результат одного прогона."""

    label: str
    config: ModelConfig
    layout: GenomeLayout
    simulator: object
    chain: EventChain
    trajectories: Trajectories
    summary: EpidemicSummary
    diagnostics: dict[str, float]
    theory: dict[str, float]
    genealogy: bool = False

    def export(self, directory: str, prefix: str | None = None) -> dict[str, str]:
        """Сохранить дерево, мутации, миграции и метаданные листьев."""
        os.makedirs(directory, exist_ok=True)
        prefix = prefix or self.label
        written: dict[str, str] = {}
        if self.genealogy:
            self.simulator.export_newick(prefix, directory)
            written["tree"] = os.path.join(directory, f"{prefix}_tree.nwk")
            self.simulator.export_mutations(f"{prefix}_mutations", directory)
            written["mutations"] = os.path.join(directory, f"{prefix}_mutations.tsv")
            if self.simulator.populations_number > 1:
                self.simulator.export_migrations(f"{prefix}_migrations", directory)
                written["migrations"] = os.path.join(directory, f"{prefix}_migrations.tsv")

            tips = tip_annotations(self.simulator, self.layout)
            path = os.path.join(directory, f"{prefix}_tips.tsv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("node\ttime\tdeme\tcourse\tgenome\n")
                for node, time, deme, state, genome in zip(
                    tips["node"], tips["time"], tips["deme"], tips["state"], tips["genome"]
                ):
                    course = HostState(int(state)).name.lower()
                    handle.write(f"{node}\t{time:.6f}\t{deme}\t{course}\t{genome}\n")
            written["tips"] = path

        path = os.path.join(directory, f"{prefix}_trajectories.csv")
        _write_trajectories(path, self.trajectories, self.config)
        written["trajectories"] = path
        return written


@dataclass
class ComparisonResult:
    """Пара прогонов и сводная таблица."""

    runs: list[RunResult] = field(default_factory=list)

    @property
    def table(self) -> list[dict]:
        return [run.summary.as_row() for run in self.runs]

    def render_table(self) -> str:
        rows = self.table
        if not rows:
            return "(пусто)"
        columns = list(rows[0])
        widths = [
            max(len(str(column)), *(len(str(row[column])) for row in rows))
            for column in columns
        ]
        header = " | ".join(str(c).ljust(w) for c, w in zip(columns, widths))
        rule = "-+-".join("-" * w for w in widths)
        body = [
            " | ".join(str(row[c]).ljust(w) for c, w in zip(columns, widths)) for row in rows
        ]
        return "\n".join([header, rule, *body])


def run_model(
    config: ModelConfig,
    *,
    seed: int = 1234,
    iterations: int = 20_000_000,
    epidemic_time: float = 400.0,
    sample_size: int | None = None,
    genealogy: bool = True,
    steps: int = 400,
    strict: bool = True,
) -> RunResult:
    """Собрать, прогнать и разобрать одну модель."""
    model = ChronicModel(config)
    simulator = model.build(seed=seed, strict=strict)
    simulator.simulate(
        iterations, sample_size=sample_size, epidemic_time=epidemic_time
    )

    chain = load_event_chain(simulator)
    trajectories = collect_trajectories(simulator, model.layout, steps=steps, chain=chain)
    summary = summarise(trajectories, config.label)
    diagnostics = bias_report(chain, model.layout, config.chronic, trajectories)

    built_genealogy = False
    if genealogy and chain.counts()["samples"] >= 2:
        simulator.genealogy(seed)
        built_genealogy = True

    return RunResult(
        label=config.label,
        config=config,
        layout=model.layout,
        simulator=simulator,
        chain=chain,
        trajectories=trajectories,
        summary=summary,
        diagnostics=diagnostics,
        theory=linearise(config).summary(),
        genealogy=built_genealogy,
    )


def compare(
    stratified: ModelConfig,
    *,
    target_r0: float = 2.5,
    seed: int = 1234,
    **kwargs,
) -> ComparisonResult:
    """Прогнать стратифицированную модель и её однородный аналог при равном R0."""
    stratified = calibrate_transmission(stratified, target_r0)
    baseline = calibrate_transmission(homogeneous_config(stratified), target_r0)
    return ComparisonResult(
        runs=[
            run_model(baseline, seed=seed, **kwargs),
            run_model(stratified, seed=seed, **kwargs),
        ]
    )


def _write_trajectories(path: str, trajectories: Trajectories, config: ModelConfig) -> None:
    names = [s.name for s in config.population.strata]
    header = ["time"]
    for deme, name in enumerate(names):
        for state in (HostState.ACUTE, HostState.CHRONIC):
            header.append(f"I_{name}_{state.name.lower()}")
    for deme, name in enumerate(names):
        for state in (HostState.ACUTE, HostState.CHRONIC):
            header.append(f"samples_{name}_{state.name.lower()}")
    header += ["I_total", "I_chronic", "chronic_share"]

    columns = [trajectories.time]
    for deme in range(len(names)):
        for state in (HostState.ACUTE, HostState.CHRONIC):
            columns.append(trajectories.infectious[deme, int(state)])
    for deme in range(len(names)):
        for state in (HostState.ACUTE, HostState.CHRONIC):
            columns.append(trajectories.sampled[deme, int(state)])
    columns += [trajectories.total, trajectories.chronic_load, trajectories.chronic_share]

    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="", fmt="%.6g")
