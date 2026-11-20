import json
import random
from pathlib import Path

from text_bereinigen import bereinige_text

# Der Ordner mit den fünf Beispieltexten.
projektordner = Path(__file__).resolve().parent.parent
textordner = projektordner / "examples" / "texte"

# Dateipfade sammeln und sortieren, damit die Ausgangsreihenfolge feststeht.
dateien = sorted(textordner.glob("*.txt"))

# Texte lesen und leere sowie doppelte Texte vor der Aufteilung aussortieren.
texte = []
bekannte_texte = {}
for datei in dateien:
    text = bereinige_text(datei.read_text(encoding="utf-8"))
    if not text:
        print(f"Übersprungen (leer): {datei.name}")
    elif text in bekannte_texte:
        print(f"Übersprungen (doppelt): {datei.name} entspricht {bekannte_texte[text]}")
    else:
        bekannte_texte[text] = datei.name
        texte.append((datei.name, text))

if len(texte) < 2:
    raise ValueError("Mindestens zwei unterschiedliche, nicht leere Texte werden benötigt.")

# Ein fester Startwert macht das Mischen wiederholbar.
zufall = random.Random(50)
zufall.shuffle(texte)

# Alle Texte bis auf den letzten zum Training, den letzten zur Validierung.
training = texte[:-1]
validierung = texte[-1:]

print("Training:")
for name, text in training:
    print(f"  {name}: {text}")

print("\nValidierung:")
for name, text in validierung:
    print(f"  {name}: {text}")

# Die Texte mit ihren Dateinamen in zwei gut lesbaren JSON-Dateien speichern.
ausgabeordner = projektordner / "data" / "processed"
ausgabeordner.mkdir(parents=True, exist_ok=True)

for dateiname, gruppe in [("training.json", training), ("validierung.json", validierung)]:
    dokumente = [{"datei": name, "text": text} for name, text in gruppe]
    json_text = json.dumps(dokumente, ensure_ascii=False, indent=2)
    ausgabedatei = ausgabeordner / dateiname
    ausgabedatei.write_text(json_text + "\n", encoding="utf-8")
    print(f"Gespeichert: {ausgabedatei}")
