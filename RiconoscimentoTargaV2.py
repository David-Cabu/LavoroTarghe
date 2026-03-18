import cv2
import warnings
from datetime import datetime
import re
import json
import os
import uuid
import subprocess
import threading
import urllib.request

warnings.filterwarnings("ignore", message=".*pin_memory.*")
import easyocr
import numpy as np

BOX_COLOR = (0, 80, 0)
TEXT_COLOR = (0, 80, 0)

# Pattern targa italiana: AA000AA oppure AA000A (vecchio formato)
PLATE_PATTERN = re.compile(r'^[A-Z]{2}\d{3}[A-Z]{2}$|^[A-Z]{2}\d{3}[A-Z]$', re.IGNORECASE)

# Percorso del database JSON (stesso folder dello script)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(REPO_DIR, "targhe_db.json")


# ============================================================
# STRUTTURA DATI PERSISTENTE
# ============================================================
#
# Il file targhe_db.json ha questa forma:
#
# {
#   "sessioni": {
#     "<session_id>": {
#       "avvio": "2026/03/18-14:00:00:000",
#       "fine":  "2026/03/18-14:05:32:147"
#     }, ...
#   },
#   "targhe": [
#     {
#       "id":         "<uuid4>",
#       "targa":      "AB123CD",
#       "timestamp":  "2026/03/18-14:32:05:347",
#       "confidenza": 0.91,
#       "sessione":   "<session_id>"
#     }, ...
#   ]
# }
#
# ============================================================

def db_load() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sessioni": {}, "targhe": []}


def db_save(db: dict) -> None:
    """Scrittura atomica: write su .tmp poi os.replace."""
    tmp = DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DB_PATH)


def db_open_session(db: dict, session_id: str, now: datetime) -> None:
    db["sessioni"][session_id] = {"avvio": format_timestamp(now), "fine": None}
    db_save(db)


def db_close_session(db: dict, session_id: str, now: datetime) -> None:
    if session_id in db["sessioni"]:
        db["sessioni"][session_id]["fine"] = format_timestamp(now)
    db_save(db)


def db_add_plate(db: dict, plate: str, ts: datetime, conf: float, session_id: str) -> dict:
    record = {
        "id":         str(uuid.uuid4()),
        "targa":      plate,
        "timestamp":  format_timestamp(ts),
        "confidenza": round(float(conf), 4),
        "sessione":   session_id
    }
    db["targhe"].append(record)
    db_save(db)
    return record


def db_query(db: dict, plate: str = None, session_id: str = None) -> list:
    results = db["targhe"]
    if plate:
        results = [r for r in results if r["targa"] == plate.upper()]
    if session_id:
        results = [r for r in results if r["sessione"] == session_id]
    return results


def db_print_summary(db: dict) -> None:
    total    = len(db["targhe"])
    unique   = len({r["targa"] for r in db["targhe"]})
    sessioni = len(db["sessioni"])
    print(f"\n{'='*52}")
    print(f"  ARCHIVIO  —  {total} rilevazioni  |  {unique} targhe uniche  |  {sessioni} sessioni")
    print(f"{'='*52}")
    for r in db["targhe"][-10:]:
        print(f"  {r['targa']}  {r['timestamp']}  conf={r['confidenza']:.2f}  sess={r['sessione'][:8]}")
    if total > 10:
        print(f"  ... e altri {total - 10} record nel file {DB_PATH}")
    print(f"{'='*52}\n")


# ============================================================
# GIT PUSH — eseguito in background, non blocca l'OCR
# ============================================================
NETLIFY_HOOK="https://api.netlify.com/build_hooks/69bb03fc1e0b859fb381ded7"
def _git_push_worker(message: str) -> None:
    """
    Esegue  git add targhe_db.json  →  git commit  →  git push
    nella directory del repo. Gira in un thread secondario.
    """
    try:
        cmds = [
            ["git", "add", "targhe_db.json"],
            ["git", "commit", "-m", message],
            ["git", "push"],
        ]
        for cmd in cmds:
            result = subprocess.run(
                cmd,
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                timeout=30
            )
            # "nothing to commit" non è un errore reale
            if result.returncode != 0 and "nothing to commit" not in result.stdout:
                print(f"  [git] ⚠  {' '.join(cmd)}: {result.stderr.strip()}")
                return
        print("  [git] ✓ push completato")
    except FileNotFoundError:
        print("  [git] ⚠  git non trovato nel PATH")
    except subprocess.TimeoutExpired:
        print("  [git] ⚠  push timeout (rete assente?)")
    except Exception as e:
        print(f"  [git] ⚠  errore: {e}")
        # trigger rebuild Netlify (~20-30 secondi e il sito è aggiornato)
    try:
        urllib.request.urlopen(
            urllib.request.Request(NETLIFY_HOOK, method='POST', data=b'')
        )
        print("  [netlify] ✓ rebuild triggerato")
    except Exception as e:
        print(f"  [netlify] ⚠ {e}")


def git_push_async(message: str) -> None:
    """Lancia il push in background senza bloccare il main loop."""
    t = threading.Thread(target=_git_push_worker, args=(message,), daemon=True)
    t.start()


# -----------------------------
# 1. OCR Reader
# -----------------------------
reader = easyocr.Reader(['it', 'en'], gpu=False)


# -----------------------------
# 2. Preprocessing modules
# -----------------------------

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def denoise(img):
    return cv2.fastNlMeansDenoising(img, None, h=6, templateWindowSize=7, searchWindowSize=5)

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    return clahe.apply(img)

def measure_sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def sharpen_if_needed(img, threshold=40):
    sharpness = measure_sharpness(img)
    if sharpness < threshold:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    return img

def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# -----------------------------
# 3. Formattazione output targa
# -----------------------------

def format_timestamp(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime(f"%Y/%m/%d-%H:%M:%S:{ms:03d}")

def clean_plate_text(text: str) -> str:
    text = text.upper().replace(" ", "").replace("-", "")
    return re.sub(r'[^A-Z0-9]', '', text)

def is_plate(text: str) -> bool:
    return bool(PLATE_PATTERN.match(text))

def format_plate_output(plate: str, dt: datetime) -> str:
    return f"{plate}-{format_timestamp(dt)}"


# -----------------------------
# 4. OCR + overlay
# -----------------------------

def run_ocr(img_gray, frame_color, db: dict, session_id: str):
    now    = datetime.now()
    result = reader.readtext(img_gray, detail=1, paragraph=False)
    base   = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    out    = base.copy()

    plates_found = []

    for (bbox, text, conf) in result:
        p1, p2, p3, p4 = bbox
        pts = np.array([p1, p2, p3, p4], dtype=int)

        cleaned     = clean_plate_text(text)
        plate_valid = is_plate(cleaned)

        color = (0, 140, 0) if plate_valid else (100, 100, 100)
        cv2.polylines(out, [pts], True, color, 1)

        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        label = cleaned if plate_valid else text
        cv2.putText(out, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        if conf > 0.3:
            if plate_valid:
                record     = db_add_plate(db, cleaned, now, conf, session_id)
                output_str = format_plate_output(cleaned, now)
                plates_found.append(output_str)
                print(f"[TARGA] {output_str}  (conf: {conf:.2f})  id={record['id'][:8]}")
                storico = db_query(db, plate=cleaned)
                if len(storico) > 1:
                    print(f"        ↳ già vista {len(storico)-1} volta/e in archivio")
            else:
                print(f"[{conf:.2f}] {text}  →  '{cleaned}' (non riconosciuta come targa)")

    if not plates_found:
        print("Nessuna targa riconosciuta in questo frame.")
    else:
        # ── push su GitHub dopo ogni scansione con nuove targhe ──
        nomi = ", ".join(p.split("-")[0] for p in plates_found)
        git_push_async(f"ocr: {nomi} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

    return out, plates_found


# -----------------------------
# 5. Webcam loop
# -----------------------------

db         = db_load()
session_id = str(uuid.uuid4())
db_open_session(db, session_id, datetime.now())

print(f"Sessione avviata: {session_id[:8]}...")
print(f"Database: {DB_PATH}  ({len(db['targhe'])} record esistenti)")
print("Premi 's' per eseguire OCR sul frame corrente")
print("Premi 'q' / 'x' per uscire")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore nella lettura della webcam")
        break

    gray  = to_gray(frame)
    den   = denoise(gray)
    cla   = apply_clahe(den)
    sharp = sharpen_if_needed(cla)
    desk  = deskew(sharp)

    cv2.imshow("1 - Originale", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("\n--- OCR ---")
        ocr_img, _ = run_ocr(desk, frame, db, session_id)
        cv2.imshow("7 - OCR Result", ocr_img)
        cv2.waitKey(0)

    if key in (ord('q'), ord('Q'), ord('x'), ord('X')):
        break

cap.release()
cv2.destroyAllWindows()

db_close_session(db, session_id, datetime.now())
# push finale per aggiornare anche la chiusura sessione
git_push_async(f"session close [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
db_print_summary(db)