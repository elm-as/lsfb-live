"""Tests unitaires pour l'indexation et le filtrage des données LSFB ISOL."""

from pathlib import Path
import json
import pandas as pd
import pytest

from lsfb_live.dataset.dataset_index import (
    load_instances_metadata,
    load_sign_occurrences,
    load_split_instance_ids,
    select_top_signs_instances,
)
from lsfb_live.dataset.downloader import (
    build_metadata_relative_paths,
    collect_target_pose_paths,
)


def test_select_top_signs_instances_filters_correctly():
    """Vérifie que la sélection conserve uniquement les N signes les plus fréquents."""
    instances_df = pd.DataFrame({
        "id": ["inst_1", "inst_2", "inst_3", "inst_4"],
        "sign": ["BONJOUR", "MERCI", "MAISON", "BONJOUR"],
        "signer": [1, 2, 3, 4],
        "start": [0, 0, 0, 0],
        "end": [10, 10, 10, 10],
    })
    occurrences_df = pd.DataFrame({
        "sign": ["BONJOUR", "MERCI", "MAISON"],
        "occurrences": [100, 50, 10],
    })

    top_2_df = select_top_signs_instances(instances_df, occurrences_df, top_k=2)
    assert set(top_2_df["sign"]) == {"BONJOUR", "MERCI"}
    assert len(top_2_df) == 3


def test_select_top_signs_handles_non_positive_k():
    """Vérifie le comportement de repli lorsque top_k est nul ou négatif."""
    instances_df = pd.DataFrame({"id": ["1"], "sign": ["A"]})
    occurrences_df = pd.DataFrame({"sign": ["A"], "occurrences": [1]})
    empty_df = select_top_signs_instances(instances_df, occurrences_df, top_k=0)
    assert empty_df.empty


def test_load_split_instance_ids_raises_on_missing_file(tmp_path: Path):
    """Vérifie qu'une exception claire est levée quand un fichier de split n'existe pas."""
    with pytest.raises(FileNotFoundError):
        load_split_instance_ids(tmp_path, split_name="unknown_split")


def test_collect_target_pose_paths_generates_all_pairs():
    """Vérifie que les chemins de poses générés couvrent chaque combinaison instance x landmark."""
    instance_ids = {"inst_A", "inst_B"}
    landmarks = ("pose", "left_hand")
    paths = collect_target_pose_paths(instance_ids, landmarks)

    expected = {
        "poses/pose/inst_A.npy",
        "poses/left_hand/inst_A.npy",
        "poses/pose/inst_B.npy",
        "poses/left_hand/inst_B.npy",
    }
    assert set(paths) == expected


def test_build_metadata_relative_paths_includes_essential_files():
    """Vérifie que la liste des métadonnées contient instances.csv et les splits principaux."""
    paths = build_metadata_relative_paths()
    assert "instances.csv" in paths
    assert "metadata/sign_occurrences.csv" in paths
    assert "metadata/splits/train.json" in paths
    assert "metadata/splits/test.json" in paths

