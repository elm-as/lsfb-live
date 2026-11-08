"""Point d'entrée pour l'entraînement, le Text-to-Sign et le Motion Capture avec Rig Humain 3D (V3).

Trois modes interactifs dans la même fenêtre scindée :
1. Mode Catalogue Corpus 3D : Le mannequin 3D rejoue les vraies trajectoires 3D LSFB avec IK.
2. Mode Text-to-Sign [T]    : Traduit une phrase française en séquence de signes 3D enchaînés.
3. Mode Motion Capture [M]  : Vos propres mouvements face webcam pilotent le mannequin 3D en direct.
"""

from pathlib import Path
import argparse
import sys
import cv2
import numpy as np

from lsfb_live.avatar.rig_skeleton import retarget_frame_landmarks_to_rig
from lsfb_live.avatar.text_to_sign_sequencer import (
    SignSequenceTimeline,
    get_sign_rig_at_progress,
    parse_french_to_lsfb_glosses,
)
from lsfb_live.capture.holistic_extractor import HolisticExtractor, ensure_holistic_model_present
from lsfb_live.capture.webcam_stream import WebcamStream
from lsfb_live.display.coach_renderer import assemble_practice_split_screen, build_coach_reference_card
from lsfb_live.display.rig_renderer_cv import render_rig_mannequin_cv
from lsfb_live.display.visualizer import draw_hud_panel, draw_skeleton_layer
from lsfb_live.recognition.live_engine import LiveInferenceEngine

GUIDED_SIGNS: list[dict[str, object]] = [
    {"sign": "AIMER", "description": "Main ouverte posee sur le coeur/torse glissant avec douceur.", "is_bimanual": False},
    {"sign": "PAPA", "description": "Main en 'L' (pouce + index tendus) tapotant pres du menton.", "is_bimanual": False},
    {"sign": "OUBLIER", "description": "Main passant devant le front et se refermant en poing.", "is_bimanual": False},
    {"sign": "COMPRENDRE", "description": "Poing pres de la tempe qui s'ouvre vers le haut en 'L'.", "is_bimanual": False},
    {"sign": "RENCONTRER", "description": "Deux index verticaux face a face se rejoignant au centre.", "is_bimanual": True},
    {"sign": "DIFFERENT", "description": "Deux index au centre qui s'ecartent vivement sur les cotes.", "is_bimanual": True},
    {"sign": "ADORER", "description": "Mains ouvertes descendant devant le torse avec intensite.", "is_bimanual": True},
    {"sign": "MAISON", "description": "Mains obliques jointes par le bout des doigts formant un toit.", "is_bimanual": True},
    {"sign": "CORPS", "description": "Mains ouvertes descendant le long du buste jusqu'a la taille.", "is_bimanual": True},
    {"sign": "AVANCER", "description": "Les deux mains ouvertes poussant franchement vers la camera.", "is_bimanual": True},
    {"sign": "MONTRER", "description": "Main droite avec l'index pointe qui vise droit devant vous.", "is_bimanual": False},
    {"sign": "INTERPRETE", "description": "Mains ouvertes avec rotation alternee de va-et-vient au torse.", "is_bimanual": True},
    {"sign": "ECOLE", "description": "Deux mains ouvertes devant le buste qui battent ensemble.", "is_bimanual": True},
    {"sign": "FAMILLE", "description": "Deux mains ouvertes decrivant un cercle horizontal devant soi.", "is_bimanual": True},
    {"sign": "AMI", "description": "Deux poings fermes qui se rejoignent chaleureusement au centre.", "is_bimanual": True},
]

TEXT_TO_SIGN_SENTENCES: list[str] = [
    "Mon papa aime toujours la famille",
    "Rencontrer un ami different a l'ecole",
    "L'interprete explique et aide a comprendre l'histoire",
    "Avancer vers la maison sans oublier",
]


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments CLI pour le mode coach et Text-to-Sign V3."""
    parser = argparse.ArgumentParser(description="Avatar Humain Rigué 3D (V3) : Coach, Text-to-Sign & Motion Capture.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index de la caméra (défaut : 0)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Seuil de validation (défaut : 0.25)")
    parser.add_argument("--sentence", type=str, default=None, help="Phrase en français à traduire en signes 3D dès l'ouverture")
    parser.add_argument("--model-path", type=Path, default=None, help="Poids du modèle (défaut: bistream_best.pt)")
    parser.add_argument("--model-type", type=str, default="bistream", choices=["bigru", "transformer", "msst", "bistream"])
    parser.add_argument("--mapping-path", type=Path, default=Path("models/class_mapping.json"))
    return parser.parse_args()


def run_coach_session(
    stream: WebcamStream,
    extractor: HolisticExtractor,
    engine: LiveInferenceEngine,
) -> None:
    """Boucle principale de la session d'entraînement avec Mannequin Humain 3D Rigué."""
    args = parse_cli_arguments()
    window_name = "LSFB V3 — Mannequin Humain Rigue 3D & Webcam Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    sentences = list(TEXT_TO_SIGN_SENTENCES)
    if args.sentence:
        sentences.insert(0, args.sentence)

    current_idx = 0
    sentence_idx = 0
    mode = "sentence" if args.sentence else "catalog"  # 'catalog', 'sentence', 'mocap'
    yaw_deg = 0.0
    cycle_steps = 45
    step_counter = 0

    try:
        while True:
            success, webcam_frame = stream.read_frame()
            if not success:
                break

            landmarks = extractor.extract_frame_landmarks(webcam_frame)
            prediction = engine.push_frame(landmarks)

            webcam_frame = cv2.flip(webcam_frame, 1)
            draw_skeleton_layer(webcam_frame, landmarks, is_mirrored=True)
            draw_hud_panel(webcam_frame, prediction, stream.fps)

            ref_canvas = np.zeros((720, 640, 3), dtype=np.uint8)
            ref_canvas[:] = (18, 21, 28)

            if mode == "mocap":
                live_rig = retarget_frame_landmarks_to_rig(landmarks)
                render_rig_mannequin_cv(ref_canvas, live_rig, yaw_deg=yaw_deg)
                sign_name = prediction.sign if prediction is not None else "RETARGETING"
                build_coach_reference_card(
                    ref_canvas,
                    f"MOCAP : {sign_name}",
                    f"Retargeting 3D en direct (Angle 3D : {yaw_deg:+.0f} deg)",
                    1.0,
                    is_bimanual=False,
                )
                is_bimanual = False
            elif mode == "sentence":
                phrase_fr = sentences[sentence_idx]
                glosses = parse_french_to_lsfb_glosses(phrase_fr)
                timeline = SignSequenceTimeline(glosses)
                seq_steps = cycle_steps * max(1, len(timeline.glosses))
                progress = float(step_counter % seq_steps) / float(seq_steps)
                step_counter += 1
                sign_name, rig_state = timeline.evaluate(progress)
                render_rig_mannequin_cv(ref_canvas, rig_state, yaw_deg=yaw_deg)
                gloss_str = " -> ".join(timeline.glosses)
                build_coach_reference_card(
                    ref_canvas,
                    f"TEXT-TO-SIGN : {sign_name}",
                    f"\"{phrase_fr}\" ({gloss_str})",
                    progress,
                    is_bimanual=False,
                )
                is_bimanual = False
            else:
                active_sign_spec = GUIDED_SIGNS[current_idx]
                sign_name = str(active_sign_spec["sign"])
                is_bimanual = bool(active_sign_spec["is_bimanual"])
                progress = float(step_counter % cycle_steps) / float(cycle_steps)
                step_counter += 1
                rig_state = get_sign_rig_at_progress(sign_name, progress)
                render_rig_mannequin_cv(ref_canvas, rig_state, yaw_deg=yaw_deg)
                build_coach_reference_card(
                    ref_canvas,
                    f"{sign_name}  ({yaw_deg:+.0f} deg)",
                    str(active_sign_spec["description"]),
                    progress,
                    is_bimanual=is_bimanual,
                )

            has_left_hand = not np.all(landmarks.left_hand == 0)
            has_right_hand = not np.all(landmarks.right_hand == 0)
            hands_warning = None
            if is_bimanual and (not has_left_hand or not has_right_hand):
                hands_warning = "ATTENTION : 2 MAINS REQUISES (Levez vos deux mains !)"

            is_match = False
            target_conf = 0.0
            if mode != "mocap" and prediction is not None and hands_warning is None:
                for cand_sign, cand_prob in prediction.top5:
                    if cand_sign == sign_name:
                        target_conf = cand_prob
                        if cand_prob >= args.threshold:
                            is_match = True
                        break

            split_screen = assemble_practice_split_screen(
                ref_canvas,
                webcam_frame,
                is_match,
                target_conf,
                hands_warning=hands_warning,
            )
            cv2.imshow(window_name, split_screen)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("t"), ord("T")):
                mode = "sentence" if mode != "sentence" else "catalog"
                step_counter = 0
            if key in (ord("m"), ord("M")):
                mode = "mocap" if mode != "mocap" else "catalog"
            if key in (ord("a"), ord("A")):
                yaw_deg = max(-45.0, yaw_deg - 10.0)
            if key in (ord("d"), ord("D")):
                yaw_deg = min(45.0, yaw_deg + 10.0)
            if key in (ord("n"), ord("N")):
                if mode == "sentence":
                    sentence_idx = (sentence_idx + 1) % len(sentences)
                else:
                    current_idx = (current_idx + 1) % len(GUIDED_SIGNS)
                step_counter = 0
            if key in (ord("p"), ord("P")):
                if mode == "sentence":
                    sentence_idx = (sentence_idx - 1) % len(sentences)
                else:
                    current_idx = (current_idx - 1) % len(GUIDED_SIGNS)
                step_counter = 0
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    """Initialise le mode coach V3 et démarre la fenêtre interactive."""
    args = parse_cli_arguments()
    ensure_holistic_model_present(Path("models/holistic_landmarker.task"))

    print("=== Démarrage de LSFB V3 (Mannequin Humain Rigué 3D) ===")
    print("Modes : [Catalogue 3D] | [T] Text-to-Sign | [M] Motion Capture Live | [A/D] Rotation 3D")

    model_type = args.model_type
    if args.model_path is not None:
        model_path = args.model_path
    elif Path("models/bistream_best.pt").is_file():
        model_path = Path("models/bistream_best.pt")
        model_type = "bistream"
    else:
        model_path = Path("models/msst_best.pt")
        model_type = "msst"

    try:
        stream = WebcamStream(camera_index=args.camera_index)
        extractor = HolisticExtractor(Path("models/holistic_landmarker.task"))
        engine = LiveInferenceEngine(
            model_path=model_path,
            mapping_path=args.mapping_path,
            model_type=model_type,
            confidence_threshold=args.threshold,
        )
    except Exception as init_error:
        print(f"Erreur d'initialisation : {init_error}", file=sys.stderr)
        return 1

    try:
        run_coach_session(stream, extractor, engine)
        return 0
    finally:
        extractor.close()
        stream.release()
        print("\nSession LSFB V3 terminee.")


if __name__ == "__main__":
    sys.exit(main())
