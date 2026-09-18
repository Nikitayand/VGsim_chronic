"""Модуль хронических пациентов для симулятора VGsim.

Субпопуляция затяжного (хронического) течения инфекции выделяется **внутри**
компартмента инфицированных, без изменения ядра VGsim: в геном добавляется
один служебный сайт, старший разряд которого делит множество гаплотипов на
два блока с независимыми интенсивностями передачи, выздоровления и
секвенирования, а переход между блоками — штатное событие мутации.

Быстрый старт::

    from vgsim_chronic import default_config, compare

    result = compare(default_config(), target_r0=2.5, epidemic_time=400)
    print(result.render_table())
"""

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
from .model import ChronicModel, build_simulator
from .params import (
    ChronicModule,
    CourseParams,
    HostPopulation,
    ImmunityModel,
    ModelConfig,
    MutationModel,
    Stratum,
    chronicity_risk_from_strata,
    default_config,
    homogeneous_config,
    homogeneous_with_duration,
    progression_rate_from_risk,
    with_chronicity_risk,
)
from .scenarios import ComparisonResult, RunResult, compare, run_model
from .theory import (
    LinearisedEpidemic,
    calibrate_transmission,
    effective_contact_matrix,
    linearise,
    mixing_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "ChronicModel",
    "ChronicModule",
    "ComparisonResult",
    "CourseParams",
    "EpidemicSummary",
    "EventChain",
    "GenomeLayout",
    "HostPopulation",
    "HostState",
    "ImmunityModel",
    "LinearisedEpidemic",
    "ModelConfig",
    "MutationModel",
    "RunResult",
    "Stratum",
    "Trajectories",
    "bias_report",
    "build_simulator",
    "calibrate_transmission",
    "chronicity_risk_from_strata",
    "collect_trajectories",
    "compare",
    "default_config",
    "effective_contact_matrix",
    "homogeneous_config",
    "homogeneous_with_duration",
    "inheritance_diagnostics",
    "linearise",
    "load_event_chain",
    "mixing_matrix",
    "progression_rate_from_risk",
    "run_model",
    "sample_table",
    "summarise",
    "with_chronicity_risk",
    "terminal_branch_lengths",
    "tip_annotations",
]
