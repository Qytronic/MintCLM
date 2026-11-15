# MintCLM

MintCLM ist ein Lernprojekt zum Bau eines eigenen **kausalen Sprachmodells** in Python und PyTorch. Das Ziel ist, Schritt für Schritt zu verstehen, wie aus Textdaten ein Modell entsteht, das das nächste Token vorhersagt und daraus Text erzeugen kann.

## Aktueller Stand

**Das Projekt befindet sich in der Planungs- und Lernphase.** Ordnerstruktur, Leitfaden und erste Platzhalter sind vorhanden. Tokenizer, Modell, Training, Evaluation und Chat-Oberfläche sind noch nicht implementiert. Es gibt daher noch kein trainiertes MintCLM-Modell und keinen funktionierenden Chat oder Trainingsbefehl.

| Bereich | Stand |
|---|---|
| Projektstruktur und Lernleitfaden | Vorhanden |
| Datenaufbereitung und Tokenizer | Geplant |
| Modell und Training | Geplant |
| Evaluation und Chat | Geplant |

## Der Plan

1. Python, NumPy und PyTorch anhand kleiner Übungen kennenlernen.
2. Textdaten auswählen, bereinigen und in Trainings- und Validierungsdaten aufteilen.
3. Einen BPE-Tokenizer bauen und testen.
4. Die Bausteine eines kleinen Transformers implementieren und überprüfen.
5. Ein erstes kleines Modell trainieren, auswerten und Text generieren lassen.
6. Später grössere Modelle, Fine-Tuning und eine Chat-Oberfläche untersuchen.

Das erste Modellziel heisst **MintCLM-1:1M** und soll ungefähr eine Million Parameter haben. Grössere Varianten bis hin zu MintCLM-1:4B sind längerfristige Ideen, keine bereits existierenden Modelle. Die tatsächlichen Grössen und Möglichkeiten hängen von der Implementierung und verfügbaren Hardware ab.

## Hier anfangen

- [Projektleitfaden](docs/project/leitfaden.md): Übungen, Erklärungen und Schritte zum ersten Modell.
- [Modellreihe und Ziele](docs/project/modellreihe.md): geplante Modelle und längerfristige Ideen.
- [Architektur](docs/architecture/design.md) und [Datenquellen](docs/data/sources.md): Platz für Entwurf und Dokumentation.

Der Leitfaden und die Modellübersicht sind auch als HTML-Dateien im Ordner `docs/project/` vorhanden. Nach dem Herunterladen kannst du sie lokal im Browser öffnen; GitHub zeigt HTML-Dateien im Repository normalerweise als Quelltext an.

## Projektordner

| Ordner | Zweck |
|---|---|
| `src/mintclm/` | Geplanter Python-Code für Daten, Tokenizer, Modell, Training und Textgenerierung |
| `scripts/` | Geplante Einstiegspunkte für die einzelnen Arbeitsschritte |
| `tests/` | Platz für Tests der einzelnen Bausteine |
| `configs/` | Einstellungen für Modelle und Experimente |
| `docs/` | Leitfaden, Planung und Dokumentation |
| `experiments/` | Versuchsbeschreibungen und kleine Ergebnisübersichten |
| `app/` | Platz für eine spätere Chat-Oberfläche |
| `data/` | Lokale Trainings- und Validierungsdaten |
| `artifacts/` | Lokale Modellstände und andere erzeugte Dateien |

Viele Dateien in diesen Ordnern sind noch leer oder Platzhalter. Git speichert keine leeren Ordner; `.gitkeep`-Dateien halten die benötigte Struktur sichtbar.

## Entwicklungsumgebung vorbereiten

Du brauchst Python, Git und eine passende PyTorch-Installation. Eine NVIDIA-GPU ist für spätere Trainingsversuche hilfreich, zum Durcharbeiten der Grundlagen aber nicht zwingend nötig.

Falls du das Projekt noch nicht auf deinem Computer hast, kannst du es zuerst herunterladen:

```powershell
git clone https://github.com/Qytronic/MintCLM.git
cd MintCLM
```

Wenn du bereits im Projektordner arbeitest, kannst du diesen Schritt überspringen.

Öffne unter Windows ein Terminal im Projektordner und erstelle eine virtuelle Umgebung:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Falls Python 3.12 fehlt, installiere es zuerst oder verwende eine mit deinen Paketen kompatible Version. Wähle anschliessend auf der [offiziellen PyTorch-Seite](https://pytorch.org/get-started/locally/) den Installationsbefehl für dein Betriebssystem und deine Hardware. Installiere danach die übrigen Projektabhängigkeiten:

```powershell
python -m pip install -r requirements.txt
```

Ob PyTorch eine verfügbare CUDA-GPU erkennt, kannst du so prüfen:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA verfügbar:', torch.cuda.is_available())"
```

Eine vorbereitete Umgebung bedeutet noch nicht, dass Training oder Tests bereits implementiert sind. Weitere Notizen stehen in [docs/training/setup.md](docs/training/setup.md).

## Änderungen auf GitHub hochladen

GitHub aktualisiert sich nicht automatisch, wenn du lokal eine Datei änderst. Prüfe deine Änderungen und erstelle dann einen neuen Speicherstand:

```powershell
git status
git add .
git status
git commit -m "Beschreibe deine Änderungen"
git push
```

Prüfe vor dem Commit, welche Dateien aufgenommen werden: Dieses Repository ist öffentlich. Lade keine Passwörter, API-Schlüssel oder privaten Daten hoch. `.gitignore` schliesst unter anderem virtuelle Umgebungen, lokale Daten und Modellstände aus; in `data/` und `artifacts/` werden nur `.gitkeep`-Platzhalter erfasst. Bereits von Git verfolgte Dateien entfernt `.gitignore` nicht nachträglich.

Für fremde Trainingsdaten und Code müssen die jeweiligen Nutzungsrechte separat geprüft und dokumentiert werden. Die Git-Regeln beeinflussen ausserdem nicht die Synchronisierung durch kDrive.

## Lizenz

Der Projektcode steht unter der [GNU Affero General Public License v3.0](LICENSE). Die Rechte und Bedingungen für später verwendete Datensätze und veröffentlichte Modellgewichte müssen jeweils gesondert geprüft und dokumentiert werden.
