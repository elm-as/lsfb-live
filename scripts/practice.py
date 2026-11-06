"""Point d'entrée pour le mode entraînement interactif (Coach LSFB).

Affiche un écran scindé : à gauche l'animation de référence d'un signeur natif,
à droite votre flux webcam avec détection et rétroaction en direct.
"""

from pathlib import Path
import argparse
import sys
import cv2
import numpy as np

from lsfb_live.capture.holistic_extractor import (
    FrameLandmarks,
    HolisticExtractor,
    ensure_holistic_model_present,
)
from lsfb_live.capture.webcam_stream import WebcamStream
from lsfb_live.display.coach_renderer import (
    assemble_practice_split_screen,
    build_coach_reference_card,
    draw_centered_skeleton,
)
from lsfb_live.display.visualizer import draw_hud_panel, draw_skeleton_layer
from lsfb_live.recognition.live_engine import LiveInferenceEngine

# Catalogue des signes guidés recommandés avec instances nettes à forte amplitude
GUIDED_SIGNS: list[dict[str, str]] = [
    {
        "sign": "RENCONTRER",
        "instance_id": "CLSFBI0505A_S012_B_141585_142879",
        "description": "Deux index verticaux qui s'avancent et se touchent face a face.",
    },
    {
        "sign": "DIFFERENT",
        "instance_id": "CLSFBI0115A_S002_B_515404_516821",
        "description": "Index pointes vers l'avant qui s'ecartent vivement sur les cotes.",
    },
    {
        "sign": "MAISON",
        "instance_id": "CLSFBI0103A_S002_B_9935_13870",
        "description": "Mains obliques jointes par le bout des doigts formant un toit pointu.",
    },
    {
        "sign": "CORPS",
        "instance_id": "CLSFBI1415A_S033_B_272212_272698",
        "description": "Mains ouvertes descendant le long du buste, de la poitrine a la taille.",
    },
    {
        "sign": "INTERPRETE",
        "instance_id": "CLSFBI0313A_S007_B_196842_202070",
        "description": "Mains face a face avec rotation alternee de va-et-vient au torse.",
    },
    {
        "sign": "AVANCER",
        "instance_id": "CLSFBI0110A_S001_B_126658_130076",
        "description": "Les deux mains ouvertes qui poussent franchement vers la camera.",
    },
    {
        "sign": "MONTRER",
        "instance_id": "CLSFBI0111A_S001_B_357087_357635",
        "description": "Index dominant tendu qui pointe franchement droit devant.",
    },
]


def interpolate_sequence(raw_poses: tuple[np.ndarray, np.ndarray, np.ndarray], target_frames: int = 45) -> list[FrameLandmarks]:
    """Interpole linéairement une séquence de poses pour une animation fluide à ~30 fps."""
    pose_arr, lh_arr, rh_arr = raw_poses
    orig_t = np.linspace(0.0, 1.0, len(pose_arr))
    target_t = np.linspace(0.0, 1.0, target_frames)

    def resample_points(points: np.ndarray, count: int) -> np.ndarray:
        clean = np.nan_to_num(points, nan=0.0)
        res = np.zeros((target_frames, count, 3), dtype=np.float32)
        for idx in range(count):
            for dim in range(3):
                res[:, idx, dim] = np.interp(target_t, orig_t, clean[:, idx, dim])
        return res

    interp_pose = resample_points(pose_arr, 33)
    interp_lh = resample_points(lh_arr, 21)
    interp_rh = resample_points(rh_arr, 21)

    return [
        FrameLandmarks(
            pose=interp_pose[step],
            left_hand=interp_lh[step],
            right_hand=interp_rh[step],
            has_pose=True,
            has_hands=True,
        )
        for step in range(target_frames)
    ]


def load_guided_sign_frames(instance_id: str, raw_dir: Path) -> list[FrameLandmarks]:
    """Charge les fichiers numpy d'une instance et retourne la séquence interpolée."""
    pose_path = raw_dir / "poses" / "pose" / f"{instance_id}.npy"
    lh_path = raw_dir / "poses" / "left_hand" / f"{instance_id}.npy"
    rh_path = raw_dir / "poses" / "right_hand" / f"{instance_id}.npy"

    pose = np.load(pose_path)
    left_hand = np.load(lh_path)
    right_hand = np.load(rh_path)
    return interpolate_sequence((pose, left_hand, right_hand), target_frames=45)


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments CLI pour le mode coach."""
    parser = argparse.ArgumentParser(description="Mode entraînement interactif avec coach LSFB.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index de la caméra (défaut : 0)")
    parser.add_argument("--threshold", type=float, default=0.30, help="Seuil de détection pour valider le signe (défaut : 0.30)")
    parser.add_argument("--model-path", type=Path, default=Path("models/msst_best.pt"), help="Poids du modèle")
    parser.add_argument("--mapping-path", type=Path, default=Path("models/class_mapping.json"), help="Fichier de mappage")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/lsfb_isol"), help="Dossier des poses brutes")
    return parser.parse_args()


def run_coach_session(
    stream: WebcamStream,
    extractor: HolisticExtractor,
    engine: LiveInferenceEngine,
    args: argparse.Namespace,
) -> None:
    """Boucle principale de la session d'entraînement avec écran scindé."""
    window_name = "LSFB Coach — Mode Entrainement Interactif"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    current_idx = 0
    ref_frames = load_guided_sign_frames(GUIDED_SIGNS[current_idx]["instance_id"], args.raw_dir)
    total_ref = len(ref_frames)
    pause_steps = 15  # Pause d'une demi-seconde à la fin du signe
    cycle_length = total_ref + pause_steps
    step_counter = 0

    try:
        while True:
            success, webcam_frame = stream.read_frame()
            if not success:
                break

            webcam_frame = cv2.flip(webcam_frame, 1)
            landmarks = extractor.extract_frame_landmarks(webcam_frame)
            prediction = engine.push_frame(landmarks)

            draw_skeleton_layer(webcam_frame, landmarks)
            draw_hud_panel(webcam_frame, prediction, stream.fps)

            # Gestion de l'animation de référence
            frame_idx = min(step_counter % cycle_length, total_ref - 1)
            current_ref_landmark = ref_frames[frame_idx]
            step_counter += 1

            ref_canvas = np.zeros((720, 640, 3), dtype=np.uint8)
            ref_canvas[:] = (22, 24, 28)  # Fond ardoise foncé
            draw_centered_skeleton(ref_canvas, current_ref_landmark)

            active_sign_spec = GUIDED_SIGNS[current_idx]
            progress = float(frame_idx) / float(total_ref - 1)
            build_coach_reference_card(
                ref_canvas,
                active_sign_spec["sign"],
                active_sign_spec["description"],
                progress,
            )

            # Vérification de la réussite du signe cible
            is_match = False
            target_conf = 0.0
            if prediction is not None:
                for cand_sign, cand_prob in prediction.top5:
                    if cand_sign == active_sign_spec["sign"]:
                        target_conf = cand_prob
                        if cand_prob >= args.threshold:
                            is_match = True
                        break

            split_screen = assemble_practice_split_screen(ref_canvas, webcam_frame, is_match, target_conf)
            cv2.imshow(window_name, split_screen)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("n"), ord("N")):  # Signe suivant
                current_idx = (current_idx + 1) % len(GUIDED_SIGNS)
                ref_frames = load_guided_sign_frames(GUIDED_SIGNS[current_idx]["instance_id"], args.raw_dir)
                total_ref = len(ref_frames)
                cycle_length = total_ref + pause_steps
                step_counter = 0
            if key in (ord("p"), ord("P")):  # Signe précédent
                current_idx = (current_idx - 1) % len(GUIDED_SIGNS)
                ref_frames = load_guided_sign_frames(GUIDED_SIGNS[current_idx]["instance_id"], args.raw_dir)
                total_ref = len(ref_frames)
                cycle_length = total_ref + pause_steps
                step_counter = 0
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    """Initialise le mode coach et démarre la fenêtre interactive."""
    args = parse_cli_arguments()
    ensure_holistic_model_present(Path("models/holistic_landmarker.task"))

    print("=== Démarrage du mode Coach LSFB ===")
    print(f"Catalogue de {len(GUIDED_SIGNS)} signes faciles charges.")
    print("Commandes dans la fenetre : [N] Suivant | [P] Precedent | [Q] Quitter")

    try:
        stream = WebcamStream(camera_index=args.camera_index)
        extractor = HolisticExtractor(Path("models/holistic_landmarker.task"))
        engine = LiveInferenceEngine(
            model_path=args.model_path,
            mapping_path=args.mapping_path,
            model_type="msst",
            confidence_threshold=args.threshold,
        )
    except Exception as init_error:
        print(f"Erreur d'initialisation : {init_error}", file=sys.stderr)
        return 1

    try:
        run_coach_session(stream, extractor, engine, args)
        return 0
    finally:
        extractor.close()
        stream.release()
        print("\nSession Coach terminee.")


if __name__ == "__main__":
    sys.exit(main())
