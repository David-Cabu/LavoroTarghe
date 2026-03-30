import cv2  # importa OpenCV: la libreria che gestisce webcam, immagini e finestre grafiche
import warnings  # importa il modulo per gestire i messaggi di avviso (warning) di Python

warnings.filterwarnings("ignore", message=".*pin_memory.*")  # silenzia un warning fastidioso di EasyOCR che non ci interessa
import easyocr  # importa EasyOCR: la libreria che legge il testo nelle immagini (OCR = riconoscimento ottico dei caratteri)
import numpy as np  # importa NumPy: serve per lavorare con matrici e array di numeri (le immagini sono matrici di pixel)
import math  # importa il modulo matematico (non viene usato in questo codice, ma era stato importato)

BOX_COLOR = (0, 80, 0)   # colore del rettangolo che circonda il testo trovato: verde scuro in formato (Blu, Verde, Rosso)
TEXT_COLOR = (0, 80, 0)  # colore del testo scritto sopra il rettangolo: stesso verde scuro

# -----------------------------
# 1. OCR Reader
# -----------------------------
reader = easyocr.Reader(
    ['it', 'en'],  # insegna al lettore OCR le lingue italiano e inglese
    gpu=False      # usa la CPU invece della GPU (più lento ma funziona su qualsiasi PC)
)


# -----------------------------
# 2. Preprocessing modules
# -----------------------------

def to_gray(frame):                              # definisce una funzione che prende un'immagine a colori
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # e la converte in scala di grigi (più semplice da analizzare)


def denoise(img):                                                                    # definisce una funzione che riduce il "rumore" (i pixel spuri) nell'immagine
    return cv2.fastNlMeansDenoising(img, None, h=6, templateWindowSize=7, searchWindowSize=5)
    # h=6: forza del filtraggio (più alto = più sfocato ma meno rumore)
    # templateWindowSize=7: dimensione del blocco di confronto pixel
    # searchWindowSize=5: area in cui cercare pixel simili


def apply_clahe(img):                                       # definisce una funzione per migliorare il contrasto locale dell'immagine
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    # clipLimit=4.0: quanto può aumentare il contrasto al massimo (evita di esagerare)
    # tileGridSize=(4,4): divide l'immagine in una griglia 4x4 e migliora ogni cella separatamente
    return clahe.apply(img)  # applica il miglioramento del contrasto e restituisce l'immagine migliorata


def measure_sharpness(img):                          # definisce una funzione che misura quanto è nitida un'immagine
    return cv2.Laplacian(img, cv2.CV_64F).var()
    # Laplaciano: un filtro matematico che evidenzia i bordi
    # .var(): calcola la varianza del risultato — se è alta l'immagine è nitida, se è bassa è sfocata


def sharpen_if_needed(img, threshold=40):   # definisce una funzione che applica la nitidezza solo se serve
    sharpness = measure_sharpness(img)      # misura la nitidezza dell'immagine

    if sharpness < threshold:  # se la nitidezza è sotto 40 (immagine sfocata)...
        kernel = np.array([[0, -1,  0],    # ...crea un filtro di sharpening (matrice 3x3)
                           [-1,  5, -1],   # il pixel centrale vale 5, i vicini valgono -1:
                           [0, -1,  0]])   # questo esalta le differenze e rende i bordi più netti
        return cv2.filter2D(img, -1, kernel)  # applica il filtro all'immagine e restituisce il risultato

    return img  # se l'immagine è già abbastanza nitida, la restituisce com'è senza modifiche


def deskew(img):  # definisce una funzione che raddrizza il testo inclinato
    coords = np.column_stack(np.where(img > 0))  # trova le coordinate di tutti i pixel non neri (cioè il testo)
    if len(coords) < 10:  # se ci sono meno di 10 pixel non neri...
        return img         # ...non ha senso raddrizzare nulla, restituisce l'immagine originale

    angle = cv2.minAreaRect(coords)[-1]  # trova il rettangolo minimo che contiene il testo e ne legge l'angolo di inclinazione

    if angle < -45:            # se l'angolo è molto negativo (testo quasi verticale)...
        angle = -(90 + angle)  # ...corregge il calcolo dell'angolo
    else:
        angle = -angle  # altrimenti inverte il segno per ottenere la rotazione corretta

    (h, w) = img.shape[:2]  # legge altezza e larghezza dell'immagine
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # calcola la matrice di rotazione attorno al centro dell'immagine
    # angle: di quanti gradi ruotare, 1.0: scala (1.0 = nessun ridimensionamento)

    return cv2.warpAffine(img, M, (w, h),              # applica la rotazione all'immagine
                          flags=cv2.INTER_CUBIC,        # usa interpolazione cubica (alta qualità)
                          borderMode=cv2.BORDER_REPLICATE)  # riempie i bordi vuoti replicando i pixel del bordo


# -----------------------------
# 3. OCR + overlay
# -----------------------------
def run_ocr(img_gray, frame_color):  # definisce la funzione principale di riconoscimento testo
    result = reader.readtext(img_gray, detail=1, paragraph=False)
    # chiede a EasyOCR di leggere il testo nell'immagine grigia
    # detail=1: restituisce anche le coordinate del testo e il livello di confidenza
    # paragraph=False: tratta ogni parola/riga separatamente invece di raggrupparle

    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # converte l'immagine grigia in BGR (serve per disegnare colori sopra)
    out = base.copy()  # crea una copia su cui disegnare, così l'originale non viene modificato

    for (bbox, text, conf) in result:  # per ogni testo trovato da EasyOCR...
        p1, p2, p3, p4 = bbox          # ...estrae i 4 angoli del rettangolo che circonda il testo
        pts = np.array([p1, p2, p3, p4], dtype=int)  # li mette in un array NumPy di interi

        cv2.polylines(out, [pts], True, BOX_COLOR, 1)
        # disegna il rettangolo (poligono) attorno al testo trovato
        # True: chiudi il poligono (unisci l'ultimo punto al primo)
        # BOX_COLOR: colore del rettangolo, 1: spessore della linea in pixel

        x_min = min(p[0] for p in pts)  # trova la coordinata X più a sinistra del rettangolo
        y_min = min(p[1] for p in pts)  # trova la coordinata Y più in alto del rettangolo
        text_pos = (x_min, y_min - 10)  # posiziona il testo 10 pixel sopra il rettangolo

        cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 1)
        # scrive il testo riconosciuto sopra il rettangolo
        # cv2.FONT_HERSHEY_SIMPLEX: font semplice e leggibile
        # 0.7: dimensione del testo, TEXT_COLOR: colore, 1: spessore

        if conf > 0.3:              # se la confidenza è superiore al 30% (EasyOCR è ragionevolmente sicuro)...
            print(f"[{conf:.2f}] {text}")  # ...stampa in console la confidenza (2 decimali) e il testo trovato

    return out  # restituisce l'immagine con i rettangoli e il testo disegnati sopra


# -----------------------------
# 4. Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)  # apre la webcam (0 = prima webcam disponibile sul PC)

print("Premi 's' per eseguire OCR sul frame corrente")  # istruzione per l'utente
print("Premi 'q' per uscire")                           # istruzione per l'utente

while True:  # loop infinito: continua finché non si preme 'q'
    ret, frame = cap.read()  # legge un frame dalla webcam
    # ret: True se la lettura è riuscita, False se c'è un errore
    # frame: l'immagine catturata in quel momento

    if not ret:                                    # se la lettura è fallita...
        print("Errore nella lettura della webcam") # ...avvisa l'utente
        break                                      # ...ed esce dal loop

    # Pipeline modulare: applica in sequenza tutti i filtri di preprocessing
    gray  = to_gray(frame)        # 1. converte in scala di grigi
    den   = denoise(gray)         # 2. riduce il rumore
    cla   = apply_clahe(den)      # 3. migliora il contrasto
    sharp = sharpen_if_needed(cla)# 4. aumenta la nitidezza se necessario
    desk  = deskew(sharp)         # 5. raddrizza il testo inclinato

    cv2.imshow("1 - Originale", frame)  # mostra la finestra con il video della webcam in tempo reale
    # le righe commentate sotto mostrerebbero le fasi intermedie del preprocessing (utili per debug)
    #  cv2.imshow("2 - Grayscale", gray)
    #  cv2.imshow("3 - Denoised", den)
    #  cv2.imshow("4 - CLAHE", cla)
    #  cv2.imshow("5 - Sharpen", sharp)
    #  cv2.imshow("6 - Deskew", desk)

    key = cv2.waitKey(1) & 0xFF
    # aspetta 1 millisecondo un tasto premuto
    # & 0xFF: operazione bit a bit per ottenere solo gli ultimi 8 bit (compatibilità cross-platform)

    if key == ord('s'):           # se l'utente ha premuto 's'...
        print("\n--- OCR ---")    # ...stampa un separatore in console
        ocr_img = run_ocr(desk, frame)           # ...esegue l'OCR sull'immagine preprocessata
        cv2.imshow("7 - OCR Result", ocr_img)    # ...mostra il risultato con i rettangoli disegnati
        cv2.waitKey(0)            # ...aspetta che l'utente prema qualsiasi tasto prima di continuare

    if key in (ord('q'), ord('Q'), ord('x'), ord('X')):  # se l'utente ha premuto q, Q, x o X...
        break                                             # ...esce dal loop

cap.release()           # rilascia la webcam (la rende disponibile ad altri programmi)
cv2.destroyAllWindows() # chiude tutte le finestre grafiche aperte da OpenCV