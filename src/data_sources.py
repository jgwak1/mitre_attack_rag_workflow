from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.schemas import (
    AnalystHeuristicSignatureConfig,
    AttackTechniqueRecord,
    PermissionsConfig,
    ThreatReport,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class LocalDataSources:
    # Validated local data loaded from the data/ directory.
    # This object gives later modules one clean place to access the corpus,
    # analyst_heuristic_signatures, sample reports, and permission settings.
    attack_techniques: list[AttackTechniqueRecord]
    analyst_heuristic_signatures: AnalystHeuristicSignatureConfig
    sample_reports: list[ThreatReport]
    permissions: PermissionsConfig


def load_json(path: Path) -> Any:
    # Small shared JSON loader.
    # Keeping this separate makes error locations easier to debug later.
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_attack_techniques(data_dir: Path = DATA_DIR) -> list[AttackTechniqueRecord]:
    # Load and validate the local ATT&CK technique corpus.
    # Each item must match AttackTechniqueRecord in schemas.py.
    raw_records = load_json(data_dir / "attack_techniques.json")
    return [AttackTechniqueRecord.model_validate(record) for record in raw_records]


def load_analyst_heuristic_signatures(data_dir: Path = DATA_DIR) -> AnalystHeuristicSignatureConfig:
    # Load and validate the analyst-curated analyst_heuristic_signatures.
    # This maps behavior phrases to candidate ATT&CK techniques.
    raw_config = load_json(data_dir / "analyst_heuristic_signatures.json")
    return AnalystHeuristicSignatureConfig.model_validate(raw_config)


def load_sample_reports(data_dir: Path = DATA_DIR) -> list[ThreatReport]:
    # Load and validate sample reports used for evaluation/regression.
    # expected_techniques are labels only and should not be used by retrieval.
    raw_reports = load_json(data_dir / "sample_reports.json")
    return [ThreatReport.model_validate(report) for report in raw_reports]


def load_permissions(data_dir: Path = DATA_DIR) -> PermissionsConfig:
    # Load and validate source/tool permission settings.
    # The schema accepts the current top-level default_user JSON shape.
    raw_permissions = load_json(data_dir / "permissions.json")
    return PermissionsConfig.model_validate(raw_permissions)


def load_all_data(data_dir: Path = DATA_DIR) -> LocalDataSources:
    # Convenience loader for later pipeline modules and tests.
    return LocalDataSources(
        attack_techniques=load_attack_techniques(data_dir),
        analyst_heuristic_signatures=load_analyst_heuristic_signatures(data_dir),
        sample_reports=load_sample_reports(data_dir),
        permissions=load_permissions(data_dir),
    )