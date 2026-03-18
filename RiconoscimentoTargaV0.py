import cv2
import warnings
from datetime import datetime
import re

warnings.filterwarnings("ignore", message=".*pin_memory.*")
import easyocr
import numpy as np

BOX_COLOR = (0, 80, 0)
TEXT_COLOR = (0, 80, 0)

# Pattern targa italiana: AA000AA oppure AA000A (vecchio formato)
PLATE_PATTERN = re.compile(r'^[A-Z]{2}\d{3}[A-Z]{2}$|^[A-Z]{2}\d{3}[A-Z]$', re.IGNORECASE)

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
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
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
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# -----------------------------
# 3. Formattazione output targa
# -----------------------------

def format_timestamp(dt: datetime) -> str:
    """
    Restituisce il timestamp nel formato AAAA/MM/GG-OO:MM:SS:ms
    dove ms sono i millisecondi a 3 cifre.
    """
    ms = dt.microsecond // 1000  # microsecondi → millisecondi
    return dt.strftime(f"%Y/%m/%d-%H:%M:%S:{ms:03d}")


def clean_plate_text(text: str) -> str:
    """
    Rimuove spazi e caratteri non alfanumerici, porta tutto in maiuscolo.
    Sostituisce i comuni errori OCR (0↔O, 1↔I, ecc.) dove il contesto
    della struttura della targa lo suggerisce.
    """
    text = text.upper().replace(" ", "").replace("-", "")
    # Solo lettere e cifre
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def is_plate(text: str) -> bool:
    """Controlla se il testo corrisponde al pattern di una targa italiana."""
    return bool(PLATE_PATTERN.match(text))


def format_plate_output(plate: str, dt: datetime) -> str:
    """
    Combina targa e timestamp nel formato richiesto:
    AA000AA-AAAA/MM/GG-OO:MM:SS:MsMsMs
    """
    return f"{plate}-{format_timestamp(dt)}"


# -----------------------------
# 4. OCR + overlay
# -----------------------------

def run_ocr(img_gray, frame_color):
    # Timestamp al momento dello scatto
    now = datetime.now()

    result = reader.readtext(img_gray, detail=1, paragraph=False)

    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    out = base.copy()

    plates_found = []

    for (bbox, text, conf) in result:
        p1, p2, p3, p4 = bbox
        pts = np.array([p1, p2, p3, p4], dtype=int)

        cleaned = clean_plate_text(text)
        plate_valid = is_plate(cleaned)

        # Colore box: verde scuro se targa valida, grigio altrimenti
        color = (0, 140, 0) if plate_valid else (100, 100, 100)

        cv2.polylines(out, [pts], True, color, 1)

        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)

        label = cleaned if plate_valid else text
        cv2.putText(out, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        if conf > 0.3:
            if plate_valid:
                output_str = format_plate_output(cleaned, now)
                plates_found.append(output_str)
                print(f"[TARGA] {output_str}  (conf: {conf:.2f})")
            else:
                print(f"[{conf:.2f}] {text}  →  '{cleaned}' (non riconosciuta come targa)")

    if not plates_found:
        print("Nessuna targa riconosciuta in questo frame.")

    return out, plates_found


# -----------------------------
# 5. Webcam loop
# -----------------------------

cap = cv2.VideoCapture(0)

print("Premi 's' per eseguire OCR sul frame corrente")
print("Premi 'q' / 'x' per uscire")

all_plates = []  # storico di tutte le targhe rilevate nella sessione

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
        ocr_img, plates = run_ocr(desk, frame)
        all_plates.extend(plates)
        cv2.imshow("7 - OCR Result", ocr_img)
        cv2.waitKey(0)

    if key in (ord('q'), ord('Q'), ord('x'), ord('X')):
        break

cap.release()
cv2.destroyAllWindows()

# Riepilogo a fine sessione
if all_plates:
    print("\n=== TARGHE RILEVATE NELLA SESSIONE ===")
    for entry in all_plates:
        print(entry)