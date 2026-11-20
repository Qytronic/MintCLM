from pathlib import Path

# Von dieser Datei aus finden wir den Projektordner und den Beispieltext.
projektordner = Path(__file__).resolve().parent.parent
textdatei = projektordner / "examples" / "erster_text.txt"

text = textdatei.read_text(encoding="utf-8")

print("Inhalt:")
print(text)
print("Anzahl Zeichen:", len(text))
print("Anzahl Wörter (nach Leerraum getrennt):", len(text.split()))
print("Anzahl Zeilen:", len(text.splitlines()))
