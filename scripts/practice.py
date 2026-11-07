"""Point d'entrée pour le mode entraînement interactif avec Avatar LSFB (Base v3).

Affiche un écran scindé : à gauche l'avatar cinématique paramétrique (100% miroir et anatomique),
à droite votre flux webcam avec détection et rétroaction de validation en direct.
"""

from pathlib import Path
import argparse
import sys
import cv2
import numpy as np

from lsfb_live.capture.holistic_extractor import HolisticExtractor, ensure_holistic_model_present
from lsfb_live.capture.webcam_stream import WebcamStream
from lsfb_live.display.coach_renderer import assemble_practice_split_screen, build_coach_reference_card
from lsfb_live.display.parametric_avatar import render_parametric_avatar_frame
from lsfb_live.display.visualizer import draw_hud_panel, draw_skeleton_layer
from lsfb_live.recognition.live_engine import LiveInferenceEngine

# Catalogue des signes canoniques avec modèle cinématique paramétrique
GUIDED_SIGNS: list[dict[str, object]] = [
    {
        "sign": "RENCONTRER",
        "description": "Deux index verticaux face a face se rejoignant au centre.",
        "is_bimanual": True,
    },
    {
        "sign": "DIFFERENT",
        "description": "Deux index au centre qui s'ecartent vivement sur les cotes.",
        "is_bimanual": True,
    },
    {
        "sign": "MAISON",
        "description": "Mains obliques jointes par le bout des doigts formant un toit.",
        "is_bimanual": True,
    },
    {
        "sign": "CORPS",
        "description": "Mains ouvertes descendant le long du buste jusqu'a la taille.",
        "is_bimanual": True,
    },
    {
        "sign": "AVANCER",
        "description": "Les deux mains ouvertes poussant franchement vers la camera.",
        "is_bimanual": True,
    },
    {
        "sign": "MONTRER",
        "description": "Main droite avec l'index pointe qui vise droit devant vous.",
        "is_bimanual": False,
    },
    {
        "sign": "INTERPRETE",
        "description": "Mains ouvertes avec rotation alternee de va-et-vient au torse.",
        "is_bimanual": True,
    },
]


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments CLI pour le mode coach."""
    parser = argparse.ArgumentParser(description="Mode entraînement interactif avec Avatar LSFB.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index de la caméra (défaut : 0)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Seuil de détection pour valider le signe (défaut : 0.25)")
    parser.add_argument("--model-path", type=Path, default=None, help="Poids du modèle (défaut: bistream_best.pt ou msst_best.pt)")
    parser.add_argument("--model-type", type=str, default="bistream", choices=["bigru", "transformer", "msst", "bistream"], help="Architecture (défaut: bistream)")
    parser.add_argument("--mapping-path", type=Path, default=Path("models/class_mapping.json"), help="Fichier de mappage")
    return parser.parse_args()


def run_coach_session(
    stream: WebcamStream,
    extractor: HolisticExtractor,
    engine: LiveInferenceEngine,
    args: argparse.Namespace,
) -> None:
    """Boucle principale de la session d'entraînement avec écran scindé."""
    window_name = "LSFB Coach — Avatar Cinematique & Webcam Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    current_idx = 0
    cycle_steps = 45  # Durée du cycle d'animation (environ 1.5s à 30 FPS)
    step_counter = 0

    try:
        while True:
            success, webcam_frame = stream.read_frame()
            if not success:
                break

            # Extraction sur trame brute non inversée (alignée sur l'anatomie et le dataset LSFB)
            landmarks = extractor.extract_frame_landmarks(webcam_frame)
            prediction = engine.push_frame(landmarks)

            # Inversion miroir uniquement pour le confort visuel d'affichage
            webcam_frame = cv2.flip(webcam_frame, 1)
            draw_skeleton_layer(webcam_frame, landmarks, is_mirrored=True)
            draw_hud_panel(webcam_frame, prediction, stream.fps)

            # Rendu de l'avatar paramétrique cinématique (100% miroir et fluide)
            active_sign_spec = GUIDED_SIGNS[current_idx]
            sign_name = str(active_sign_spec["sign"])
            is_bimanual = bool(active_sign_spec["is_bimanual"])

            progress = float(step_counter % cycle_steps) / float(cycle_steps)
            step_counter += 1

            ref_canvas = np.zeros((720, 640, 3), dtype=np.uint8)
            ref_canvas[:] = (22, 25, 32)
            render_parametric_avatar_frame(ref_canvas, sign_name, progress)
            build_coach_reference_card(
                ref_canvas,
                sign_name,
                str(active_sign_spec["description"]),
                progress,
                is_bimanual=is_bimanual,
            )

            # Analyse des mains de l'utilisateur
            has_left_hand = not np.all(landmarks.left_hand == 0)
            has_right_hand = not np.all(landmarks.right_hand == 0)
            hands_warning = None
            if is_bimanual and (not has_left_hand or not has_right_hand):
                hands_warning = "ATTENTION : 2 MAINS REQUISES (Levez vos deux mains !)"

            # Vérification de la réussite du signe cible
            is_match = False
            target_conf = 0.0
            if prediction is not None and hands_warning is None:
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
            if key in (ord("n"), ord("N")):  # Signe suivant
                current_idx = (current_idx + 1) % len(GUIDED_SIGNS)
                step_counter = 0
            if key in (ord("p"), ord("P")):  # Signe précédent
                current_idx = (current_idx - 1) % len(GUIDED_SIGNS)
                step_counter = 0
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    """Initialise le mode coach et démarre la fenêtre interactive."""
    args = parse_cli_arguments()
    ensure_holistic_model_present(Path("models/holistic_landmarker.task"))

    print("=== Démarrage du mode Coach LSFB (Avatar Paramétrique v3) ===")
    print(f"Catalogue de {len(GUIDED_SIGNS)} signes canoniques charges.")
    print("Commandes dans la fenetre : [N] Suivant | [P] Precedent | [Q] Quitter")

    # Sélection automatique du meilleur modèle disponible
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
        run_coach_session(stream, extractor, engine, args)
        return 0
    finally:
        extractor.close()
        stream.release()
        print("\nSession Coach terminee.")


if __name__ == "__main__":
    sys.exit(main())
