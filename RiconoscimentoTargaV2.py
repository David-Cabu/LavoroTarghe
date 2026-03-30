import cv2                      # importa OpenCV: gestisce webcam, immagini e finestre grafiche
import warnings                 # serve per silenziare messaggi di avviso fastidiosi
from datetime import datetime   # importa solo la classe datetime per lavorare con data e ora
import re                       # importa il modulo per le espressioni regolari (pattern di testo)
import json                     # importa il modulo per leggere e scrivere file JSON
import os                       # importa il modulo per gestire file e cartelle del sistema operativo
import uuid                     # importa il modulo per generare ID univoci casuali
import subprocess               # importa il modulo per eseguire comandi di sistema (es. git)
import threading                # importa il modulo per eseguire operazioni in parallelo (thread)
import urllib.request           # importa il modulo per fare richieste HTTP (es. chiamare Netlify)

warnings.filterwarnings("ignore", message=".*pin_memory.*")
# silenzia un warning di PyTorch/EasyOCR che appare sulla CPU e non indica nessun errore reale

import easyocr   # importa EasyOCR: la libreria che riconosce il testo nelle immagini tramite AI
import numpy as np  # importa NumPy: le immagini sono matrici di numeri, NumPy le gestisce

BOX_COLOR = (0, 80, 0)   # colore verde scuro per il rettangolo attorno alla targa (formato BGR)
TEXT_COLOR = (0, 80, 0)  # stesso verde scuro per il testo scritto sopra il rettangolo

# espressione regolare che descrive il formato di una targa italiana:
# AA000AA (formato attuale) oppure AA000A (vecchio formato a 6 caratteri)
PLATE_PATTERN = re.compile(r'^[A-Z]{2}\d{3}[A-Z]{2}$|^[A-Z]{2}\d{3}[A-Z]$', re.IGNORECASE)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# trova il percorso assoluto della cartella in cui si trova questo file Python

DB_PATH = os.path.join(REPO_DIR, "targhe_db.json")
# costruisce il percorso completo del file database, mettendolo nella stessa cartella dello script


# ============================================================
# STRUTTURA DATI PERSISTENTE
# ============================================================
# (commento descrittivo della struttura del JSON — non è codice eseguibile)


def db_load() -> dict:
    # funzione che carica il database dal file JSON sul disco
    if os.path.exists(DB_PATH):
        # se il file esiste già...
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
            # ...aprilo in lettura e trasformalo da testo JSON a dizionario Python
    return {"sessioni": {}, "targhe": []}
    # se il file non esiste ancora, restituisce un database vuoto con la struttura corretta


def db_save(db: dict) -> None:
    # funzione che salva il dizionario Python come file JSON sul disco
    tmp = DB_PATH + ".tmp"
    # crea un percorso temporaneo: "targhe_db.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
        # scrive il dizionario nel file temporaneo in formato JSON leggibile (indent=2 = rientri)
    os.replace(tmp, DB_PATH)
    # sostituisce il file reale con quello temporaneo in modo atomico:
    # se il programma crasha durante la scrittura, il file originale rimane intatto


def db_open_session(db: dict, session_id: str, now: datetime) -> None:
    # registra l'avvio di una nuova sessione nel database
    db["sessioni"][session_id] = {"avvio": format_timestamp(now), "fine": None}
    # aggiunge una voce con l'ora di avvio; "fine" è None perché la sessione è ancora aperta
    db_save(db)
    # salva subito il database su disco


def db_close_session(db: dict, session_id: str, now: datetime) -> None:
    # registra la chiusura della sessione corrente
    if session_id in db["sessioni"]:
        # solo se questa sessione esiste davvero nel database...
        db["sessioni"][session_id]["fine"] = format_timestamp(now)
        # ...aggiorna il campo "fine" con l'ora attuale
    db_save(db)
    # salva il database aggiornato su disco


def db_add_plate(db: dict, plate: str, ts: datetime, conf: float, session_id: str) -> dict:
    # funzione che aggiunge una nuova targa rilevata al database
    record = {
        "id":         str(uuid.uuid4()),   # genera un ID univoco casuale per questo record
        "targa":      plate,               # la stringa della targa (es. "AB123CD")
        "timestamp":  format_timestamp(ts),# la data e ora della rilevazione formattata
        "confidenza": round(float(conf), 4),# il livello di certezza dell'OCR, arrotondato a 4 decimali
        "sessione":   session_id           # collega questo record alla sessione corrente
    }
    db["targhe"].append(record)
    # aggiunge il record in fondo alla lista delle targhe
    db_save(db)
    # salva immediatamente su disco (ogni targa viene persistita all'istante)
    return record
    # restituisce il record appena creato (serve per stampare l'ID in console)


def db_query(db: dict, plate: str = None, session_id: str = None) -> list:
    # funzione di ricerca nel database: filtra per targa e/o sessione
    results = db["targhe"]
    # parte dall'intera lista di targhe
    if plate:
        results = [r for r in results if r["targa"] == plate.upper()]
        # se è specificata una targa, tieni solo i record con quella targa (in maiuscolo)
    if session_id:
        results = [r for r in results if r["sessione"] == session_id]
        # se è specificata una sessione, tieni solo i record di quella sessione
    return results
    # restituisce la lista filtrata


def db_print_summary(db: dict) -> None:
    # stampa in console un riepilogo finale dell'archivio a fine sessione
    total    = len(db["targhe"])                    # conta tutte le rilevazioni
    unique   = len({r["targa"] for r in db["targhe"]})  # conta le targhe uniche (set elimina i duplicati)
    sessioni = len(db["sessioni"])                  # conta il numero di sessioni registrate
    print(f"\n{'='*52}")
    print(f"  ARCHIVIO  —  {total} rilevazioni  |  {unique} targhe uniche  |  {sessioni} sessioni")
    print(f"{'='*52}")
    for r in db["targhe"][-10:]:
        # mostra gli ultimi 10 record ([-10:] = "prendi gli ultimi 10 elementi della lista")
        print(f"  {r['targa']}  {r['timestamp']}  conf={r['confidenza']:.2f}  sess={r['sessione'][:8]}")
        # stampa targa, timestamp, confidenza e i primi 8 caratteri dell'ID sessione
    if total > 10:
        print(f"  ... e altri {total - 10} record nel file {DB_PATH}")
        # se ci sono più di 10 record totali, avvisa che gli altri sono nel file
    print(f"{'='*52}\n")


# ============================================================
# GIT PUSH — eseguito in background, non blocca l'OCR
# ============================================================

NETLIFY_HOOK = "https://api.netlify.com/build_hooks/69bb03fc1e0b859fb381ded7"
# URL del webhook Netlify: chiamarlo con POST avvia un rebuild del sito

def _git_push_worker(message: str) -> None:
    # funzione che esegue i comandi git; il "_" davanti al nome indica che è "privata"
    try:
        cmds = [
            ["git", "add", "targhe_db.json"],  # mette il file nel prossimo commit
            ["git", "commit", "-m", message],  # crea il commit con il messaggio passato
            ["git", "push"],                   # manda tutto su GitHub
        ]
        for cmd in cmds:
            # esegue i tre comandi uno alla volta
            result = subprocess.run(
                cmd,
                cwd=REPO_DIR,        # esegue il comando nella cartella del progetto
                capture_output=True, # cattura output e errori invece di stamparli
                text=True,           # restituisce output come stringa invece che bytes
                timeout=30           # se dopo 30 secondi non ha finito, considera fallito
            )
            if result.returncode != 0 and "nothing to commit" not in result.stdout:
                # se il comando è fallito E non è perché "non c'è nulla da committare"...
                print(f"  [git] ⚠  {' '.join(cmd)}: {result.stderr.strip()}")
                return
                # ...stampa l'errore e interrompi (non tentare il comando successivo)
        print("  [git] ✓ push completato")
        # se tutti e tre i comandi sono andati bene, stampa conferma

    except FileNotFoundError:
        print("  [git] ⚠  git non trovato nel PATH")
        # git non è installato o non è nel PATH di sistema
    except subprocess.TimeoutExpired:
        print("  [git] ⚠  push timeout (rete assente?)")
        # il push ha impiegato più di 30 secondi (probabilmente niente internet)
    except Exception as e:
        print(f"  [git] ⚠  errore: {e}")
        # qualsiasi altro errore imprevisto

    try:
        urllib.request.urlopen(
            urllib.request.Request(NETLIFY_HOOK, method='POST', data=b'')
        )
        # manda una richiesta POST vuota all'URL del webhook di Netlify
        # questo dice a Netlify "aggiorna il sito adesso"
        print("  [netlify] ✓ rebuild triggerato")
    except Exception as e:
        print(f"  [netlify] ⚠ {e}")
        # se la chiamata a Netlify fallisce, avvisa ma non blocca nulla


def git_push_async(message: str) -> None:
    # funzione pubblica che lancia il push in un thread separato
    t = threading.Thread(target=_git_push_worker, args=(message,), daemon=True)
    # crea un thread che eseguirà _git_push_worker; daemon=True significa che
    # se il programma principale termina, questo thread viene fermato automaticamente
    t.start()
    # avvia il thread: da questo momento gira in parallelo senza bloccare la webcam


# -----------------------------
# 1. OCR Reader
# -----------------------------
reader = easyocr.Reader(['it', 'en'], gpu=False)
# inizializza il motore OCR con i modelli italiano e inglese, usando la CPU


# -----------------------------
# 2. Preprocessing modules
# -----------------------------

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # converte l'immagine da colori (BGR) a scala di grigi: più semplice da analizzare per l'OCR

def denoise(img):
    return cv2.fastNlMeansDenoising(img, None, h=6, templateWindowSize=7, searchWindowSize=5)
    # riduce il rumore digitale dell'immagine (i pixel "sporchi" causati dalla webcam)
    # h=6: intensità del filtro; valori più alti sfocano di più ma rimuovono più rumore

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    # CLAHE = miglioramento adattivo del contrasto: divide l'immagine in una griglia 4×4
    # e migliora il contrasto di ogni cella separatamente, evitando sovraesposizioni
    return clahe.apply(img)
    # applica il miglioramento e restituisce l'immagine con contrasto migliorato

def measure_sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()
    # misura la nitidezza: applica il filtro Laplaciano (che evidenzia i bordi)
    # e ne calcola la varianza — alta varianza = immagine nitida, bassa = sfocata

def sharpen_if_needed(img, threshold=40):
    sharpness = measure_sharpness(img)
    # misura quanto è nitida l'immagine
    if sharpness < threshold:
        # se la nitidezza è sotto 40 (immagine sfocata)...
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # ...crea un filtro 3×3: il pixel centrale vale 5, i vicini -1
        # questo esalta le differenze tra pixel adiacenti rendendo i bordi più netti
        return cv2.filter2D(img, -1, kernel)
        # applica il filtro all'immagine e restituisce il risultato
    return img
    # se è già sufficientemente nitida, non fare nulla e restituiscila com'è

def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    # trova le coordinate (riga, colonna) di tutti i pixel non neri (cioè il testo)
    if len(coords) < 10:
        return img
        # se ci sono meno di 10 pixel visibili non ha senso calcolare l'angolo, esci subito
    angle = cv2.minAreaRect(coords)[-1]
    # calcola il rettangolo minimo che contiene tutti i pixel del testo
    # e prende solo l'angolo di inclinazione ([-1] = ultimo elemento della tupla restituita)
    if angle < -45:
        angle = -(90 + angle)
        # correzione matematica per angoli molto negativi (testo quasi verticale)
    else:
        angle = -angle
        # inverte il segno per ottenere la direzione di rotazione corretta
    (h, w) = img.shape[:2]
    # legge altezza e larghezza dell'immagine
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # calcola la matrice matematica per ruotare attorno al centro esatto dell'immagine
    # 1.0 = nessun ridimensionamento
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # applica la rotazione con qualità alta (INTER_CUBIC)
    # i bordi vuoti lasciati dalla rotazione vengono riempiti replicando i pixel del bordo


# -----------------------------
# 3. Formattazione output targa
# -----------------------------

def format_timestamp(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    # converte i microsecondi in millisecondi (divide per 1000)
    return dt.strftime(f"%Y/%m/%d-%H:%M:%S:{ms:03d}")
    # formatta la data come "2026/03/18-14:32:05:347" (:03d = millisecondi sempre a 3 cifre)

def clean_plate_text(text: str) -> str:
    text = text.upper().replace(" ", "").replace("-", "")
    # porta tutto in maiuscolo e rimuove spazi e trattini
    return re.sub(r'[^A-Z0-9]', '', text)
    # rimuove qualsiasi carattere che non sia una lettera o un numero
    # (es. punto, parentesi, simboli vari che l'OCR può leggere per errore)

def is_plate(text: str) -> bool:
    return bool(PLATE_PATTERN.match(text))
    # controlla se il testo rispetta il formato targa italiana (AA000AA o AA000A)
    # restituisce True se è una targa valida, False altrimenti

def format_plate_output(plate: str, dt: datetime) -> str:
    return f"{plate}-{format_timestamp(dt)}"
    # combina la targa e il timestamp nel formato finale: "AB123CD-2026/03/18-14:32:05:347"


# -----------------------------
# 4. OCR + overlay
# -----------------------------

def run_ocr(img_gray, frame_color, db: dict, session_id: str):
    # funzione principale: esegue l'OCR e disegna i risultati sull'immagine
    now    = datetime.now()
    # registra l'ora esatta in cui viene eseguita la scansione
    result = reader.readtext(img_gray, detail=1, paragraph=False)
    # chiede a EasyOCR di leggere tutto il testo nell'immagine
    # detail=1: restituisce anche coordinate e confidenza per ogni testo trovato
    # paragraph=False: ogni riga/parola è un risultato separato, non raggruppato
    base   = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # converte l'immagine grigia in BGR così possiamo disegnarci sopra a colori
    out    = base.copy()
    # crea una copia su cui disegnare, lasciando intatta l'immagine originale

    plates_found = []
    # lista che raccoglierà le stringhe "TARGA-TIMESTAMP" delle targhe valide trovate

    for (bbox, text, conf) in result:
        # per ogni testo trovato dall'OCR: bbox=coordinate, text=testo, conf=confidenza
        p1, p2, p3, p4 = bbox
        # estrae i 4 angoli del rettangolo che circonda il testo
        pts = np.array([p1, p2, p3, p4], dtype=int)
        # li mette in un array NumPy di interi (OpenCV richiede interi per disegnare)

        cleaned     = clean_plate_text(text)
        # pulisce il testo rimuovendo spazi e caratteri strani
        plate_valid = is_plate(cleaned)
        # controlla se il testo pulito è una targa italiana valida

        color = (0, 140, 0) if plate_valid else (100, 100, 100)
        # verde se è una targa valida, grigio se non lo è
        cv2.polylines(out, [pts], True, color, 1)
        # disegna il rettangolo attorno al testo sull'immagine di output

        x_min = min(p[0] for p in pts)
        # trova la coordinata X più a sinistra del rettangolo
        y_min = min(p[1] for p in pts)
        # trova la coordinata Y più in alto del rettangolo
        label = cleaned if plate_valid else text
        # usa il testo pulito se è una targa, altrimenti il testo originale dell'OCR
        cv2.putText(out, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        # scrive il testo 10 pixel sopra il rettangolo con font standard, dimensione 0.7

        if conf > 0.3:
            # procede solo se la confidenza dell'OCR è superiore al 30%
            if plate_valid:
                record     = db_add_plate(db, cleaned, now, conf, session_id)
                # salva la targa nel database e ottieni il record appena creato
                output_str = format_plate_output(cleaned, now)
                # costruisce la stringa "AB123CD-2026/03/18-14:32:05:347"
                plates_found.append(output_str)
                # aggiunge la stringa alla lista delle targhe trovate in questa scansione
                print(f"[TARGA] {output_str}  (conf: {conf:.2f})  id={record['id'][:8]}")
                # stampa in console la targa, la confidenza e i primi 8 caratteri dell'ID
                storico = db_query(db, plate=cleaned)
                # cerca nel database quante volte questa targa è già stata vista
                if len(storico) > 1:
                    print(f"        ↳ già vista {len(storico)-1} volta/e in archivio")
                    # se è già stata vista in precedenza, lo comunica (len-1 perché esclude quella appena aggiunta)
            else:
                print(f"[{conf:.2f}] {text}  →  '{cleaned}' (non riconosciuta come targa)")
                # testo trovato ma non è una targa: stampalo comunque per debug

    if not plates_found:
        print("Nessuna targa riconosciuta in questo frame.")
        # se la lista è vuota, nessuna targa valida è stata trovata in questa scansione
    else:
        nomi = ", ".join(p.split("-")[0] for p in plates_found)
        # estrae solo i numeri di targa dalla lista (es. "AB123CD, EF456GH")
        git_push_async(f"ocr: {nomi} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
        # lancia il push su GitHub in background con un messaggio di commit descrittivo

    return out, plates_found
    # restituisce l'immagine con i rettangoli disegnati e la lista delle targhe trovate


# -----------------------------
# 5. Webcam loop
# -----------------------------

db         = db_load()
# carica il database esistente dal disco (o crea uno vuoto se non esiste)
session_id = str(uuid.uuid4())
# genera un ID univoco per questa sessione (es. "a3f7c2d1-...")
db_open_session(db, session_id, datetime.now())
# registra l'avvio della sessione nel database con l'ora attuale

print(f"Sessione avviata: {session_id[:8]}...")
# stampa solo i primi 8 caratteri dell'ID sessione (abbastanza per identificarla)
print(f"Database: {DB_PATH}  ({len(db['targhe'])} record esistenti)")
# informa l'utente del percorso del database e quante targhe ci sono già
print("Premi 's' per eseguire OCR sul frame corrente")
print("Premi 'q' / 'x' per uscire")

cap = cv2.VideoCapture(0)
# apre la webcam (0 = prima webcam disponibile sul sistema)

while True:
    # loop infinito: continua a girare finché l'utente non preme q o x
    ret, frame = cap.read()
    # legge un singolo frame dalla webcam
    # ret = True se la lettura è riuscita, False se c'è un problema
    # frame = l'immagine acquisita in quel momento
    if not ret:
        print("Errore nella lettura della webcam")
        break
        # se la lettura fallisce, avvisa e interrompe il loop

    gray  = to_gray(frame)         # 1. converte in scala di grigi
    den   = denoise(gray)          # 2. riduce il rumore
    cla   = apply_clahe(den)       # 3. migliora il contrasto
    sharp = sharpen_if_needed(cla) # 4. aumenta la nitidezza se necessario
    desk  = deskew(sharp)          # 5. raddrizza il testo inclinato

    cv2.imshow("1 - Originale", frame)
    # mostra la finestra con il video della webcam in tempo reale (non il frame processato)

    key = cv2.waitKey(1) & 0xFF
    # aspetta 1 millisecondo un tasto premuto
    # & 0xFF: operazione bit a bit per compatibilità su tutti i sistemi operativi

    if key == ord('s'):
        # se l'utente ha premuto il tasto 's'...
        print("\n--- OCR ---")
        ocr_img, _ = run_ocr(desk, frame, db, session_id)
        # ...esegue l'OCR sul frame preprocessato; _ ignora la lista targhe (già gestita dentro)
        cv2.imshow("7 - OCR Result", ocr_img)
        # mostra la finestra con i rettangoli e il testo disegnati
        cv2.waitKey(0)
        # aspetta che l'utente prema qualsiasi tasto prima di tornare al video in diretta

    if key in (ord('q'), ord('Q'), ord('x'), ord('X')):
        break
        # se l'utente preme q, Q, x o X esce dal loop (maiuscolo e minuscolo entrambi)

cap.release()
# rilascia la webcam rendendola disponibile ad altri programmi
cv2.destroyAllWindows()
# chiude tutte le finestre grafiche aperte da OpenCV

db_close_session(db, session_id, datetime.now())
# registra la chiusura della sessione con l'ora esatta di uscita
git_push_async(f"session close [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
# lancia un ultimo push in background per salvare anche la chiusura sessione su GitHub
db_print_summary(db)
# stampa il riepilogo finale dell'archivio in console prima di terminare