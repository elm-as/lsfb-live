"""Point d'entrée principal pour la reconnaissance de signes LSFB en direct via webcam."""

from pathlib import Path
import argparse
import sys
import cv2

from lsfb_live.capture.holistic_extractor import (
    HolisticExtractor,
    ensure_holistic_model_present,
)
from lsfb_live.capture.webcam_stream import WebcamStream
from lsfb_live.display.visualizer import draw_hud_panel, draw_skeleton_layer
from lsfb_live.recognition.live_engine import LiveInferenceEngine


def parse_cli_arguments() -> argparse.Namespace:
    """Analyse les arguments CLI pour l'application live."""
    parser = argparse.ArgumentParser(description="Traduction en direct de signes LSFB par webcam.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/msst_best.pt"),
        help="Poids du modèle entraîné (défaut : models/msst_best.pt)",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("models/class_mapping.json"),
        help="Fichier de mappage classes-signes (défaut : models/class_mapping.json)",
    )
    parser.add_argument(
        "--holistic-model",
        type=Path,
        default=Path("models/holistic_landmarker.task"),
        help="Modèle MediaPipe Holistic (défaut : models/holistic_landmarker.task)",
    )
    parser.add_argument(
        "--model-type",
        choices=["bigru", "transformer", "msst"],
        default="msst",
        help="Architecture du réseau (défaut : msst)",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Index de la webcam (défaut : 0)")
    parser.add_argument("--threshold", type=float, default=0.40, help="Seuil de confiance minimal (défaut : 0.40)")
    return parser.parse_args()


def run_live_recognition_loop(
    stream: WebcamStream,
    extractor: HolisticExtractor,
    engine: LiveInferenceEngine,
) -> None:
    """Boucle temps réel de capture, prédiction et affichage."""
    window_name = "LSFB Live — Reconnaissance en direct (Appuyez sur 'Q' pour quitter)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            success, frame = stream.read_frame()
            if not success:
                break

            # Inversion miroir pour une expérience naturelle face webcam
            frame = cv2.flip(frame, 1)

            landmarks = extractor.extract_frame_landmarks(frame)
            prediction = engine.push_frame(landmarks)

            draw_skeleton_layer(frame, landmarks)
            draw_hud_panel(frame, prediction, stream.fps)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' ou touche Échap
                break
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    """Initialise les composants matériels et logiciels et lance l'affichage."""
    args = parse_cli_arguments()

    if not args.model_path.is_file():
        print(f"Modèle introuvable : {args.model_path}. Lancez d'abord scripts/train.py", file=sys.stderr)
        return 1

    mapping_target = args.mapping_path
    if not mapping_target.is_file():
        fallback = Path("data/processed/class_mapping.json")
        if fallback.is_file():
            mapping_target = fallback
        else:
            print(f"Mappage introuvable : {mapping_target}", file=sys.stderr)
            return 1

    ensure_holistic_model_present(args.holistic_model)

    print("=== Démarrage de LSFB Live ===")
    print(f"Modèle : {args.model_path.name} ({args.model_type})")
    print(f"Mappage : {mapping_target.name}")
    print(f"Caméra index : {args.camera_index}")

    try:
        stream = WebcamStream(camera_index=args.camera_index)
        extractor = HolisticExtractor(args.holistic_model)
        engine = LiveInferenceEngine(
            model_path=args.model_path,
            mapping_path=mapping_target,
            model_type=args.model_type,
            confidence_threshold=args.threshold,
        )
    except Exception as init_error:
        print(f"Échec d'initialisation : {init_error}", file=sys.stderr)
        return 1

    try:
        run_live_recognition_loop(stream, extractor, engine)
        return 0
    finally:
        extractor.close()
        stream.release()
        print("\nSession LSFB Live terminée proprement.")


if __name__ == "__main__":
    sys.exit(main())

