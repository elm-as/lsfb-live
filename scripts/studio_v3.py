"""Serveur local pour le Studio Web 3D (Three.js) du Mannequin Humain Rigué LSFB V3.

Lance une interface WebGL 3D interactive sur http://127.0.0.1:8765 permettant :
1. Text-to-Sign : Traduire n'importe quelle phrase française en séquence 3D LSFB.
2. Corpus 3D    : Explorer sous 360° les 30 signes natifs rigués (IK + 42 phalanges).
3. MoCap Live   : Piloter le mannequin 3D en temps réel par webcam (--enable-webcam).
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import json
import sys
import webbrowser

from lsfb_live.avatar.rig_skeleton import RigSkeleton3D, retarget_frame_landmarks_to_rig
from lsfb_live.avatar.text_to_sign_sequencer import (
    SignSequenceTimeline,
    get_sign_rig_at_progress,
    load_sign_prototypes_3d,
    parse_french_to_lsfb_glosses,
)


def serialize_rig_skeleton(rig: RigSkeleton3D) -> dict[str, object]:
    """Convertit un RigSkeleton3D en dictionnaire JSON compact pour WebGL."""
    return {
        "head_center": rig.head_center.round(4).tolist(),
        "nose_tip": rig.nose_tip.round(4).tolist(),
        "neck": rig.neck.round(4).tolist(),
        "pelvis": rig.pelvis.round(4).tolist(),
        "left_shoulder": rig.left_shoulder.round(4).tolist(),
        "right_shoulder": rig.right_shoulder.round(4).tolist(),
        "left_elbow": rig.left_elbow.round(4).tolist(),
        "right_elbow": rig.right_elbow.round(4).tolist(),
        "left_wrist": rig.left_wrist.round(4).tolist(),
        "right_wrist": rig.right_wrist.round(4).tolist(),
        "left_hand": rig.left_hand.round(4).tolist(),
        "right_hand": rig.right_hand.round(4).tolist(),
    }


class StudioV3RequestHandler(BaseHTTPRequestHandler):
    """Gestionnaire HTTP léger servant le Studio Three.js et l'API cinématique 3D."""

    html_path = Path("src/lsfb_live/display/web_studio/index.html")
    mocap_provider = None

    def log_message(self, format: str, *args: object) -> None:
        return  # Silence les logs HTTP par trame

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            content = self.html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/signs":
            protos = load_sign_prototypes_3d()
            signs = sorted(protos.keys()) if protos else ["AIMER", "PAPA", "RENCONTRER", "DIFFERENT"]
            self._send_json({"signs": signs})
            return

        if parsed.path == "/api/translate":
            raw_text = query.get("text", [""])[0]
            glosses = parse_french_to_lsfb_glosses(raw_text)
            self._send_json({"glosses": glosses})
            return

        if parsed.path == "/api/pose":
            gloss_str = query.get("glosses", ["AIMER"])[0]
            progress = float(query.get("progress", ["0.0"])[0])
            glosses = [g.strip() for g in gloss_str.split(",") if g.strip()]
            timeline = SignSequenceTimeline(glosses)
            sign_name, rig = timeline.evaluate(progress)
            self._send_json({"sign": sign_name, "rig": serialize_rig_skeleton(rig)})
            return

        if parsed.path == "/api/mocap":
            if self.mocap_provider is not None:
                self._send_json(self.mocap_provider())
            else:
                rig = get_sign_rig_at_progress("AIMER", 0.5)
                self._send_json({"sign": "WEBCAM OFF (--enable-webcam)", "confidence": 0.0, "rig": serialize_rig_skeleton(rig)})
            return

        self.send_error(404)


def main() -> int:
    parser = argparse.ArgumentParser(description="Studio Web 3D Three.js pour le Mannequin Humain Rigué LSFB V3.")
    parser.add_argument("--port", type=int, default=8765, help="Port HTTP local (défaut : 8765)")
    parser.add_argument("--enable-webcam", action="store_true", help="Active la capture webcam pour le mode MoCap Live")
    parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir automatiquement le navigateur")
    args = parser.parse_args()

    stream = None
    extractor = None
    if args.enable_webcam:
        from lsfb_live.capture.holistic_extractor import HolisticExtractor, ensure_holistic_model_present
        from lsfb_live.capture.webcam_stream import WebcamStream
        from lsfb_live.recognition.live_engine import LiveInferenceEngine

        ensure_holistic_model_present(Path("models/holistic_landmarker.task"))
        stream = WebcamStream(camera_index=0)
        extractor = HolisticExtractor(Path("models/holistic_landmarker.task"))
        engine = LiveInferenceEngine(
            model_path=Path("models/bistream_best.pt"),
            mapping_path=Path("models/class_mapping.json"),
            model_type="bistream",
            confidence_threshold=0.40,
        )

        def poll_webcam() -> dict[str, object]:
            ok, frame = stream.read_frame()
            if not ok:
                rig = get_sign_rig_at_progress("AIMER", 0.0)
                return {"sign": "ERREUR CAM", "confidence": 0.0, "rig": serialize_rig_skeleton(rig)}
            lm = extractor.extract_frame_landmarks(frame)
            pred = engine.push_frame(lm)
            rig = retarget_frame_landmarks_to_rig(lm)
            return {
                "sign": pred.sign if pred else "Initialisation...",
                "confidence": pred.confidence if pred else 0.0,
                "rig": serialize_rig_skeleton(rig),
            }

        StudioV3RequestHandler.mocap_provider = staticmethod(poll_webcam)

    url = f"http://127.0.0.1:{args.port}"
    print(f"=== Studio Web 3D LSFB V3 démarré sur {url} ===")
    print("Appuyez sur Ctrl+C pour arrêter le serveur.")
    if not args.no_browser:
        webbrowser.open(url)

    server = HTTPServer(("127.0.0.1", args.port), StudioV3RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if extractor is not None:
            extractor.close()
        if stream is not None:
            stream.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
