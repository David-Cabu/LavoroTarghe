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
# NUOVE COSTANTI — OPZIONI AGGIUNTIVE
# ============================================================

# costo in euro per ogni ora di sosta nel parcheggio
# esempio: 2.0 = 2€/ora → 30 minuti costano 1€
TARIFFA_PER_ORA = 2.0

# ogni quanti secondi il programma esegue la scansione OCR in automatico, senza premere tasti
# valore basso = più reattivo ma usa più CPU; 5 secondi è un buon compromesso
OCR_INTERVAL_SECONDI = 5

# quanti secondi devono passare prima di poter rilevare DI NUOVO la stessa targa
# serve per evitare che una targa ferma davanti alla telecamera venga aggiunta 100 volte
# 30 secondi = dopo mezzo minuto la stessa targa può essere registrata un'altra volta
COOLDOWN_TARGA_SECONDI = 30


# ============================================================
# STRUTTURA DATI PERSISTENTE
# ============================================================
# Il database JSON ora ha questa struttura aggiornata:
# {
#   "sessioni":          { <session_id>: {avvio, fine} },
#   "targhe":            [ {id, targa, timestamp, confidenza, sessione} ],   ← storico OCR grezzo
#   "utenti_registrati": [ "AB123CD", "EF456GH" ],                           ← targhe autorizzate
#   "parcheggio":        { "AB123CD": { "ingresso": "2026/..." } },           ← veicoli attualmente dentro
#   "transiti":          [ {id, targa, ingresso, uscita, durata_minuti, pedaggio_euro, autorizzato} ]
# }


def db_load() -> dict:
    # funzione che carica il database dal file JSON sul disco
    if os.path.exists(DB_PATH):
        # se il file esiste già...
        with open(DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        # MIGRAZIONE: se il database è vecchio (creato prima di questa versione),
        # aggiunge i nuovi campi che mancano senza perdere i dati già salvati.
        # setdefault(chiave, valore) aggiunge la chiave SOLO se non esiste già.
        db.setdefault("utenti_registrati", [])
        # lista delle targhe pre-autorizzate; vuota di default, aggiungila a mano nel JSON
        db.setdefault("parcheggio", {})
        # dizionario dei veicoli attualmente dentro: chiave = targa, valore = {ingresso: timestamp}
        db.setdefault("transiti", [])
        # lista di tutti i transiti completati (ingresso + uscita) con il pedaggio calcolato
        return db

    # se il file non esiste ancora, restituisce un database vuoto con TUTTA la struttura corretta
    return {
        "sessioni":          {},
        "targhe":            [],
        "utenti_registrati": [],   # aggiungi qui le targhe autorizzate, es: ["AB123CD", "XY789ZZ"]
        "parcheggio":        {},   # popolato automaticamente quando una targa entra
        "transiti":          []    # popolato automaticamente quando una targa esce
    }


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

    # --- NUOVO: mostra anche i veicoli rimasti dentro e i transiti completati ---

    dentro = db.get("parcheggio", {})
    # recupera il dizionario dei veicoli attualmente dentro (può essere vuoto)
    if dentro:
        print(f"\n  VEICOLI ANCORA DENTRO ({len(dentro)}):")
        for targa, info in dentro.items():
            print(f"    {targa}  ingresso: {info['ingresso']}")
            # avvisa che questi veicoli sono entrati ma non sono ancora usciti

    transiti = db.get("transiti", [])
    # recupera la lista dei transiti completati (ingresso + uscita con pedaggio)
    if transiti:
        totale_incassato = sum(t["pedaggio_euro"] for t in transiti)
        # somma tutti i pedaggi per ottenere il totale incassato nella sessione
        print(f"\n  TRANSITI COMPLETATI: {len(transiti)}  |  INCASSATO: €{totale_incassato:.2f}")
        for t in transiti[-5:]:
            # mostra solo gli ultimi 5 transiti per non intasare la console
            print(f"    {t['targa']}  durata: {t['durata_minuti']}min  pedaggio: €{t['pedaggio_euro']:.2f}")

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


def parse_timestamp(ts: str) -> datetime:
    # NUOVA funzione: converte un timestamp salvato nel database (stringa) in un oggetto datetime
    # è l'operazione inversa di format_timestamp: da "2026/03/18-14:32:05:347" a datetime
    main, ms_str = ts.rsplit(':', 1)
    # divide la stringa all'ULTIMA ':' per separare i millisecondi dal resto
    # esempio: "2026/03/18-14:32:05:347" → main="2026/03/18-14:32:05", ms_str="347"
    dt = datetime.strptime(main, "%Y/%m/%d-%H:%M:%S")
    # trasforma la parte principale in un oggetto datetime (senza millisecondi)
    return dt.replace(microsecond=int(ms_str) * 1000)
    # aggiunge i millisecondi convertiti in microsecondi (datetime lavora in microsecondi)


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


# ============================================================
# NUOVE FUNZIONI — GESTIONE TRANSITI, BARRIERA E PEDAGGIO
# ============================================================

def gestisci_transito(db: dict, plate: str, now: datetime) -> dict:
    # NUOVA funzione: decide se la targa sta ENTRANDO o USCENDO dal parcheggio
    # e salva il transito nel database.
    #
    # Logica semplice: se la targa NON è nel dizionario "parcheggio" → sta entrando.
    #                  se la targa È già nel dizionario "parcheggio" → sta uscendo.
    # Questo funziona perché ogni targa può essere dentro UNA volta sola alla volta.

    if plate not in db["parcheggio"]:
        # ── CASO INGRESSO ──────────────────────────────────────────
        # la targa non risulta dentro → registriamo l'ingresso adesso
        db["parcheggio"][plate] = {
            "ingresso": format_timestamp(now)
            # salviamo solo il timestamp di ingresso; il resto lo calcoliamo all'uscita
        }
        db_save(db)
        # salva subito: il sito web deve vedere il veicolo entrare in tempo reale
        return {"tipo": "ingresso", "targa": plate, "timestamp": format_timestamp(now)}
        # restituisce un dizionario con le info dell'ingresso (usato da run_ocr per stampare)

    else:
        # ── CASO USCITA ─────────────────────────────────────────────
        # la targa risulta già dentro → calcola quanto tempo ha sostato e il pedaggio

        ingresso_str = db["parcheggio"][plate]["ingresso"]
        # recupera il timestamp di ingresso salvato (è una stringa, es. "2026/04/20-10:00:00:000")

        try:
            ingresso_dt = parse_timestamp(ingresso_str)
            # converte la stringa in datetime per poter fare la differenza con l'ora attuale
            durata_secondi = (now - ingresso_dt).total_seconds()
            # calcola quanti secondi sono passati dall'ingresso a adesso
            durata_minuti  = round(durata_secondi / 60, 1)
            # converte in minuti, arrotondato a 1 decimale (es. 45.5 minuti)
            durata_ore     = durata_secondi / 3600
            # converte in ore per moltiplicare con la tariffa oraria
            pedaggio       = round(durata_ore * TARIFFA_PER_ORA, 2)
            # pedaggio = ore × tariffa; round a 2 decimali = centesimi di euro
        except Exception as e:
            # se il parsing del timestamp fallisce per qualsiasi motivo (corruzione dati, ecc.)
            print(f"  [pedaggio] ⚠ impossibile calcolare la durata: {e}")
            durata_minuti = 0.0
            pedaggio      = 0.0
            # in caso di errore il pedaggio è 0 (meglio che bloccare il programma)

        # costruisce il record del transito completato da salvare nello storico
        transito = {
            "id":             str(uuid.uuid4()),         # ID univoco di questo transito
            "targa":          plate,                     # numero di targa
            "ingresso":       ingresso_str,              # timestamp di ingresso (già stringa)
            "uscita":         format_timestamp(now),     # timestamp di uscita (adesso)
            "durata_minuti":  durata_minuti,             # quanto ha sostato in minuti
            "pedaggio_euro":  pedaggio,                  # quanto deve pagare in euro
            "autorizzato":    plate in db["utenti_registrati"]
            # True se la targa è nella lista degli abbonati/autorizzati
        }

        db["transiti"].append(transito)
        # aggiunge il transito completato alla lista storica dei transiti
        del db["parcheggio"][plate]
        # rimuove la targa dal parcheggio: adesso risulta fuori
        db_save(db)
        # salva tutto su disco immediatamente

        return {"tipo": "uscita", **transito}
        # restituisce le info dell'uscita (il ** "spacchetta" il dizionario transito)


def simula_apertura_barriera(plate: str) -> None:
    # NUOVA funzione: simula l'apertura del cancello/barriera per le targhe autorizzate.
    # In un sistema reale, qui si invierebbe un segnale digitale (es. GPIO su Raspberry Pi,
    # una chiamata HTTP a un relay di rete, o un comando seriale a un Arduino).
    # In questa versione simuliamo semplicemente stampando un messaggio in console.
    print(f"  ╔══════════════════════════════════════╗")
    print(f"  ║    BARRIERA APERTA                 ║")
    print(f"  ║  Targa autorizzata: {plate:<14}      ║")
    print(f"  ║  Accesso consentito                  ║")
    print(f"  ╚══════════════════════════════════════╝")
    # qui potresti aggiungere, ad esempio:
    # import requests; requests.post("http://192.168.1.50/apri")  ← relay di rete


# ============================================================
# LOCK PER L'OCR AUTOMATICO
# ============================================================
# Problema: EasyOCR su CPU è lento (2-5 secondi per frame).
# Se girassimo l'OCR dentro il loop principale il video si bloccherebbe ogni 5 secondi.
# Soluzione: l'OCR gira in un THREAD SEPARATO così il video rimane fluido.
# Il lock serve a evitare che due scansioni girino contemporaneamente (cosa che causerebbe
# problemi perché entrambe leggerebbero e scriverebbero il database nello stesso momento).

ocr_lock = threading.Lock()
# il lock è come un semaforo: solo un thread alla volta può "tenerlo";
# se un thread cerca di prenderlo mentre è già occupato, aspetta finché non si libera.

ocr_result_img = [None]
# lista con un solo elemento usata come "contenitore condiviso" tra il thread OCR e il loop principale.
# Usiamo una lista invece di una variabile semplice perché le liste in Python sono mutabili:
# il thread può modificare ocr_result_img[0] e il loop principale vede la modifica.


# -----------------------------
# 4. OCR + overlay
# -----------------------------

def run_ocr(img_gray, frame_color, db: dict, session_id: str, cooldown_targhe: dict):
    # funzione principale: esegue l'OCR e disegna i risultati sull'immagine.
    # AGGIORNATA rispetto alla versione precedente con: cooldown, ingresso/uscita, barriera.
    #
    # Parametro nuovo: cooldown_targhe — dizionario {targa: datetime_ultima_rilevazione}
    # serve a tenere traccia di quando ogni targa è stata rilevata l'ultima volta,
    # così evitiamo di registrarla 50 volte se rimane ferma davanti alla telecamera.

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

        if conf > 0.3 and plate_valid:
            # procede solo se la confidenza è superiore al 30% E il formato è una targa valida

            # ── CONTROLLO COOLDOWN ─────────────────────────────────────────────────
            # controlla se questa targa è già stata rilevata troppo di recente
            ultima_vista = cooldown_targhe.get(cleaned)
            # get restituisce None se la targa non è mai stata vista, altrimenti il datetime

            if ultima_vista is not None:
                secondi_passati = (now - ultima_vista).total_seconds()
                # calcola quanti secondi sono passati dall'ultima rilevazione
                if secondi_passati < COOLDOWN_TARGA_SECONDI:
                    # la targa è stata vista troppo di recente: la ignoriamo per evitare duplicati
                    print(f"  [cooldown] {cleaned} ignorata "
                          f"(vista {secondi_passati:.0f}s fa, cooldown {COOLDOWN_TARGA_SECONDI}s)")
                    continue
                    # "continue" salta il resto del ciclo for e passa alla prossima targa OCR

            # ── NUOVA RILEVAZIONE VALIDA ───────────────────────────────────────────
            # arrivati qui, la targa è nuova (o il cooldown è scaduto): la processiamo

            cooldown_targhe[cleaned] = now
            # aggiorna il dizionario con l'ora di questa rilevazione

            # 1. salva nel database storico grezzo (come nella versione precedente)
            record     = db_add_plate(db, cleaned, now, conf, session_id)
            output_str = format_plate_output(cleaned, now)
            plates_found.append(output_str)
            print(f"[TARGA] {output_str}  (conf: {conf:.2f})  id={record['id'][:8]}")

            storico = db_query(db, plate=cleaned)
            if len(storico) > 1:
                print(f"        ↳ già vista {len(storico)-1} volta/e in archivio")

            # 2. gestisci ingresso o uscita dal parcheggio
            transito = gestisci_transito(db, cleaned, now)
            # questa funzione decide automaticamente se è un ingresso o un'uscita

            if transito["tipo"] == "ingresso":
                # ── disegna il box BLU per l'ingresso ──
                print(f"  [INGRESSO] {cleaned} entrato alle {now.strftime('%H:%M:%S')}")
                cv2.polylines(out, [pts], True, (200, 80, 0), 3)
                # colore BGR (200, 80, 0) = blu-azzurro; spessore 3 per renderlo visibile
                cv2.putText(out, "INGRESSO", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 80, 0), 2)
                # scrive "INGRESSO" sopra il rettangolo

            else:
                # ── disegna il box ARANCIONE per l'uscita e mostra il pedaggio ──
                durata = transito["durata_minuti"]
                pedaggio = transito["pedaggio_euro"]
                print(f"  [USCITA]   {cleaned} — durata: {durata} min — "
                      f"pedaggio: €{pedaggio:.2f}")
                cv2.polylines(out, [pts], True, (0, 140, 255), 3)
                # colore BGR (0, 140, 255) = arancione; spessore 3
                cv2.putText(out, f"USCITA  EUR {pedaggio:.2f}",
                            (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                # scrive sull'immagine "USCITA  EUR 1.50" (o qualsiasi cifra)
                cv2.putText(out, f"{durata} min",
                            (x_min, y_min - 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                # scrive anche la durata sopra

            # 3. controlla se la targa è nella lista degli utenti autorizzati
            if cleaned in db.get("utenti_registrati", []):
                # "utenti_registrati" è una lista nel database: contiene le targhe pre-autorizzate
                # es. abbonati mensili, dipendenti, residenti, ecc.
                simula_apertura_barriera(cleaned)
                # chiama la funzione che simula l'apertura del cancello

                # disegna una cornice VERDE BRILLANTE spessa per indicare l'autorizzazione
                cv2.polylines(out, [pts], True, (0, 255, 0), 4)
                # spessore 4 = ben visibile; sovrascrive i colori ingresso/uscita
                cv2.putText(out, "AUTORIZZATO", (x_min, y_min - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # scrive "AUTORIZZATO" in verde sopra tutto il resto

        elif conf > 0.3 and not plate_valid:
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


def _ocr_thread_worker(img_gray, frame_color, db: dict, session_id: str,
                       cooldown_targhe: dict) -> None:

    # viene usato un thread perchè EasyOCR su CPU impiega 2-5 secondi per analizzare un frame.
    # Se chiamassimo run_ocr() direttamente nel loop principale, il video si congelrebbe
    # per tutto quel tempo. Con il thread, l'OCR gira in parallelo e il video rimane fluido.

    with ocr_lock:
        # "with ocr_lock" è come entrare in un bagno con la serratura:
        # se qualcuno è già dentro (OCR in corso), aspetti fuori finché non esce.
        # Questo evita che due scansioni girino contemporaneamente.
        ocr_img, _ = run_ocr(img_gray, frame_color, db, session_id, cooldown_targhe)
        # esegue la scansione OCR vera e propria (può richiedere qualche secondo)
        ocr_result_img[0] = ocr_img
        # salva il risultato nel contenitore condiviso; il loop principale lo leggerà


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
print(f"Utenti registrati: {db.get('utenti_registrati', [])}")
# mostra la lista delle targhe autorizzate al momento dell'avvio
print(f"Tariffa: €{TARIFFA_PER_ORA:.2f}/ora  |  OCR automatico ogni {OCR_INTERVAL_SECONDI}s")
# informa l'utente della tariffa e della frequenza di scansione
print("Premi 's' per forzare una scansione immediata")
print("Premi 'q' / 'x' per uscire")

cap = cv2.VideoCapture(0)
# apre la webcam (0 = prima webcam disponibile sul sistema)

# ── variabili per la gestione del tempo e del cooldown ──────────────────────
ultimo_ocr    = datetime.now()
# memorizza quando è stata eseguita l'ultima scansione OCR (automatica o manuale)
# lo usiamo per sapere quando è il momento di farne un'altra

cooldown_targhe: dict = {}
# dizionario che tiene traccia dell'ultima rilevazione per ogni targa
# formato: { "AB123CD": datetime(2026, 4, 20, 10, 0, 30), ... }
# serve al meccanismo di cooldown per evitare duplicati ravvicinati

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

    # ── overlay informativo sul frame in diretta ─────────────────────────────
    ora_attuale          = datetime.now()
    secondi_al_prossimo  = OCR_INTERVAL_SECONDI - (ora_attuale - ultimo_ocr).total_seconds()
    # calcola quanti secondi mancano alla prossima scansione automatica

    frame_display = frame.copy()
    # copia il frame originale per disegnarci sopra senza modificare quello usato dall'OCR

    veicoli_dentro = len(db.get("parcheggio", {}))
    # conta quanti veicoli sono attualmente nel parcheggio (quante chiavi ha il dizionario)

    # scrive nell'angolo in alto a sinistra del video alcune informazioni utili
    cv2.putText(frame_display,
                f"Dentro: {veicoli_dentro} veicoli",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 118), 2)
    # "Dentro: 3 veicoli" — verde brillante (0, 230, 118) in formato BGR

    cv2.putText(frame_display,
                f"OCR tra: {max(0.0, secondi_al_prossimo):.1f}s",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 90), 1)
    # "OCR tra: 3.2s" — verde più scuro; max(0,x) evita di mostrare numeri negativi

    ocr_in_corso = ocr_lock.locked()
    # controlla se il thread OCR sta girando in questo momento
    if ocr_in_corso:
        cv2.putText(frame_display, "OCR IN CORSO...", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        # avvisa visivamente che la scansione è in esecuzione

    cv2.imshow("1 - Originale", frame_display)
    # mostra la finestra con il video della webcam in tempo reale (con le info sovrapposte)

    # ── mostra il risultato OCR se è pronto ──────────────────────────────────
    if ocr_result_img[0] is not None:
        cv2.imshow("7 - OCR Result", ocr_result_img[0])
        # mostra la finestra con i rettangoli disegnati dall'ultimo OCR
        # non chiamiamo waitKey(0) così la finestra rimane aperta ma il video continua

    key = cv2.waitKey(1) & 0xFF
    # aspetta 1 millisecondo un tasto premuto
    # & 0xFF: operazione bit a bit per compatibilità su tutti i sistemi operativi

    # ── OCR AUTOMATICO ogni N secondi ────────────────────────────────────────
    secondi_trascorsi = (ora_attuale - ultimo_ocr).total_seconds()
    if secondi_trascorsi >= OCR_INTERVAL_SECONDI and not ocr_lock.locked():
        # condizioni per partire: abbastanza tempo è passato E nessun OCR è già in corso
        ultimo_ocr = ora_attuale
        # aggiorna subito l'ora dell'ultimo OCR (così il countdown riparte da zero)
        print(f"\n--- OCR AUTOMATICO [{ora_attuale.strftime('%H:%M:%S')}] ---")
        t = threading.Thread(
            target=_ocr_thread_worker,
            args=(desk.copy(), frame.copy(), db, session_id, cooldown_targhe),
            daemon=True
            # daemon=True: se il programma principale termina, questo thread si ferma da solo
        )
        t.start()
        # avvia il thread: da questo momento l'OCR gira in parallelo senza bloccare il video

    # ── OCR MANUALE con 's' ───────────────────────────────────────────────────
    if key == ord('s'):
        # se l'utente ha premuto il tasto 's'...
        if not ocr_lock.locked():
            # ...ma solo se non c'è già un OCR in corso
            print("\n--- OCR MANUALE ---")
            ultimo_ocr = ora_attuale
            # resetta anche il timer automatico così non parte subito dopo
            t = threading.Thread(
                target=_ocr_thread_worker,
                args=(desk.copy(), frame.copy(), db, session_id, cooldown_targhe),
                daemon=True
            )
            t.start()
            # avvia il thread manuale (stesso meccanismo dell'automatico)
        else:
            print("  [skip] OCR già in corso, attendi...")
            # informa l'utente che deve aspettare che la scansione corrente finisca

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