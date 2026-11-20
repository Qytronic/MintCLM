# Lernbereich

Hier liegen Übungen, mit denen wir die Grundlagen für MintCLM ausprobieren.
Sie sind nicht die offizielle Daten- oder Trainingspipeline des Modells.

- `examples/`: Beispieltexte, darunter `texte/` für die Aufteilungsübung.
- `scripts/`: kleine Lernprogramme.
- `data/processed/`: lokal erzeugte Ergebnisse der Aufteilungsübung; diese werden nicht auf GitHub hochgeladen.

Starte die Programme im Terminal aus dem Projektordner `MintCLM 1.0`:

```powershell
.\.venv\Scripts\python.exe Lernbereich\scripts\text_untersuchen.py
.\.venv\Scripts\python.exe Lernbereich\scripts\texte_aufteilen.py
```

Die Aufteilungsübung überschreibt bei jedem Start ihre beiden JSON-Dateien im
Lernbereich. Die Beispieltexte bleiben unverändert.
