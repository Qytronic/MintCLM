# MintCLM Schritt für Schritt entwickeln und verstehen

**Aktualisierte Projektplanung:** Dein erstes Modell heisst **MintCLM-1:1M**. Später sind 100M oder 300M und langfristig 4B mögliche Ziele. Die [Modellreihe und Hardwareplanung](modellreihe.md) ergänzt diesen Lernleitfaden; sie ersetzt nicht die Übungen unten. Alle Modelle sind bisher geplant, nicht fertig implementiert.

Dein Ziel ist ein eigenes kleines Sprachmodell, dessen Bestandteile du erklären, prüfen und verändern kannst. Du darfst unterwegs entscheiden, wie weit du gehen möchtest. Ein funktionierender kleiner Transformer mit sorgfältiger Auswertung ist bereits ein sinnvolles technisches Projektziel. Ob und wie deine Schule dieses Ergebnis für MINT Plus F anerkennt, klärst du mit dem MINT-Team.

**Dein erstes Etappenziel:** Ein kurzer Text wird zu Zahlen, ein kleines Modell lernt daraus, und du kannst neue Zeichen erzeugen sowie den Lernfortschritt zeigen. BPE, grosse Datenmengen und zusätzliche Hardware kommen danach.

Lies heute die Übersicht, die Grundlagen und die Einrichtung. Die übrigen Kapitel sind dein Nachschlagewerk. Du musst nicht alles verstehen, bevor du anfängst. Die erste Arbeitssitzung steht am Ende.

Stand der Installationshinweise und der lokalen Bestandsaufnahme: **14. November 2026**. Die Codebeispiele sind kleine Lernbausteine. Wo ein Beispiel spätere Funktionen wie `model(...)` voraussetzt, steht das ausdrücklich dabei. Sie bilden noch kein fertiges Trainingsprogramm.

## Übersicht über deinen Weg

| Reihenfolge | Mindestziel | Später möglich | Warum dieser Schritt jetzt kommt |
|---|---|---|---|
| 0 Einrichtung | Python, eigene Umgebung, erste Rechnung | GPU und weitere Geräte | Du brauchst eine verlässliche Arbeitsumgebung. |
| 1 Grundlagen | Arrays, Tensoren, Gradienten verstehen | Vertiefte Mathematik | Damit kannst du die späteren Rechenschritte erklären. |
| 2 Textdaten | Kleine Texte untersuchen und getrennt aufteilen | Grösserer eigener Korpus | Daten bestimmen, was überhaupt lernbar ist. |
| 3 Einfache Tokens | Text in Zeichen-IDs oder Bytes umwandeln und zurück | Eigener BPE-Tokenizer | Das Modell verarbeitet Zahlen. |
| 4 Trainingsbeispiele | Eingabe und nächstes Token korrekt verschieben | Datensätze effizient laden | Ein falsches Lernziel macht jedes Training wertlos. |
| 5 Erstes Sprachmodell | Bigram-Modell als überschaubare Vorstufe | Vergleichsmodell behalten | Du verstehst Training ohne die ganze Transformer-Komplexität. |
| 6 Eigener Transformer | Maskierte Attention und ein kleiner Decoder | Mehr Schichten und andere Bausteine | Jetzt nutzt dein Modell mehrere frühere Tokens. |
| 7 Training | Einen kleinen Batch lernen, dann getrennt validieren | Lernratenplan, Mixed Precision | Zuerst die Lernfähigkeit prüfen, dann länger trainieren. |
| 8 Ausgabe und Speichern | Text erzeugen, bewerten, speichern und laden | Saubere Fortsetzung langer Läufe | Du erhältst ein nutzbares und überprüfbares Ergebnis. |
| 9 BPE | Häufige Byte-Paare selbst zusammenführen | Effizientere Implementierung | Du verbesserst einen bereits funktionierenden Ablauf. |
| 10 Erweiterungen | Nur bei einer klaren Frage erweitern | Memmap, RoPE, GQA, SFT, Oberfläche | Jede neue Idee lässt sich mit deinem Ausgangsmodell vergleichen. |

Die Kapitel zu **Tests**, **Zeit und Hardware**, **Dokumentation** und **deinem konkreten Anfang** ergänzen diese Phasen. Tests und Journal begleiten dich ab dem ersten Tag.

Du kannst BPE früher lernen, wenn dich dieser Algorithmus besonders interessiert. Es ist aber keine Voraussetzung für dein erstes trainiertes Sprachmodell. Ein Zeichenmodell ist ein vollwertiges erstes CLM.

### So findest du dich im Leitfaden zurecht

- **Für heute:** [Grundidee](#was-du-eigentlich-baust), [Einrichtung](#phase-0-deine-arbeitsumgebung-einrichten) und [erste Arbeitssitzung](#dein-konkreter-anfang).
- **Für deine ersten Bausteine:** [Grundlagen](#phase-1-die-benötigten-grundlagen-lernen), [Textdaten](#phase-2-kleine-textdaten-auswählen-und-aufteilen), [Tokens](#phase-3-eine-einfache-tokenisierung-bauen) und [Trainingsbeispiele](#phase-4-die-richtigen-trainingsbeispiele-erzeugen).
- **Für dein erstes lernendes Modell:** [Bigram](#phase-5-ein-winziges-erstes-sprachmodell-bauen), [Transformer](#phase-6-deinen-eigenen-kleinen-transformer-entwickeln), [Training](#phase-7-den-trainingsprozess-verstehen-und-durchführen) sowie [Generierung und Speichern](#phase-8-text-erzeugen-und-trainingsstände-erhalten).
- **Für spätere Ideen:** [Eigene BPE](#phase-9-bpe-schritt-für-schritt-selbst-implementieren) sowie [Memmap und Erweiterungen](#phase-10-speicher-und-architektur-gezielt-erweitern).
- **Während des ganzen Projekts:** [Tests](#tests-und-fehlersuche-als-teil-deiner-arbeit), [Zeit und Hardware](#zeit-und-hardware-realistisch-planen) und [Dokumentation mit Journalvorlage](#dein-mint-projekt-dokumentieren).

Pfadangaben wie `src/mintclm/model/attention.py` bezeichnen im Folgenden die Stelle innerhalb deines Repositorys `C:\Users\wunde\MintCLM\MintCLM 1.0`. Manche zusätzliche Übungsdateien legst du erst im jeweiligen Schritt an. Absolute Pfade für grosse Arbeitsdateien sind Beispiele für deinen Laptop und müssen auf anderen Geräten angepasst werden.

## Was du eigentlich baust

### Sprachmodell und causal

Ein Sprachmodell schätzt, welche Fortsetzung zu einem Text passt. Nach «Heute scheint die» könnte es beispielsweise «Sonne» wahrscheinlicher finden als «Schublade». Es erhält keine Liste fertiger Antworten. Seine Vorhersagen entstehen aus gelernten Zahlen und dem eingegebenen Text.

**Causal** bedeutet hier: Für die nächste Vorhersage darf das Modell nur den bisherigen Text verwenden. Es darf das gesuchte nächste Token und den restlichen zukünftigen Text nicht sehen. Der Begriff meint hier diese zeitliche Richtung, nicht ein Verständnis von Ursache und Wirkung.

Stell dir den Text «MINT!» vor. Bei einem Token pro Zeichen entstehen Lernaufgaben wie:

| Bisheriger Text | Gesuchtes nächstes Zeichen |
|---|---|
| M | I |
| MI | N |
| MIN | T |
| MINT | ! |

Ein Transformer kann solche Vorhersagen im Training für viele Positionen gleichzeitig berechnen. Eine **Maske** sperrt dabei den Blick nach rechts. Beim Erzeugen neuer Texte geht es dagegen Token für Token weiter, weil das nächste Token noch nicht feststeht.

### Die wichtigsten Begriffe

| Begriff | Verständliche Bedeutung |
|---|---|
| Token | Ein Textbaustein: zunächst ein Zeichen oder Byte, später vielleicht ein Wortteil. |
| Token-ID | Die ganze Zahl, mit der du einen Token bezeichnet hast. Eine grössere ID bedeutet keine stärkere Bedeutung. |
| Vokabular | Die Menge aller möglichen Token-IDs und ihrer Bedeutungen. Seine Grösse nennen wir `V`. |
| Parameter | Veränderbare Zahlen im Modell, zum Beispiel Einträge in einer Gewichtsmatrix. |
| Hyperparameter | Von dir gewählte Einstellungen, etwa Anzahl Schichten oder Lernrate. Sie werden im normalen Training nicht automatisch gelernt. |
| Training | Wiederholt vorhersagen, Fehler messen und Parameter ein kleines Stück anpassen. |
| Loss | Eine Zahl, die bewertet, wie schlecht die Vorhersage zum richtigen Ziel passt. Kleiner ist für dieselbe Aufgabe besser. |
| Gradient | Zeigt für jeden Parameter, wie sich der Loss bei einer kleinen Änderung ungefähr verändert. |
| Optimierer | Ein Verfahren, das aus Gradienten Parameteränderungen berechnet. Du verwendest später AdamW. |
| Batch | Mehrere Trainingsbeispiele, die zusammen verarbeitet werden. |
| Kontext | Die Tokens, die das Modell für eine Vorhersage berücksichtigen kann. |
| Validierung | Prüfung an zurückgehaltenen Texten, auf denen die Parameter nicht trainiert werden. |
| Inferenz | Anwenden des gelernten Modells. Textgenerierung ist eine Form davon. |
| Checkpoint | Gespeicherter Modell- oder Trainingsstand. |

Für einen einzelnen Zieltoken kann der Loss als `-log(p_richtig)` verstanden werden. Gibt dein Modell dem richtigen Token Wahrscheinlichkeit 0,5, beträgt dieser Loss ungefähr 0,693; bei 0,1 ungefähr 2,303. Eine selbstsichere falsche Vorhersage wird stark bestraft. Die Funktion `log` ist der natürliche Logarithmus; du musst ihn zunächst nur als Teil dieser Fehlerbewertung verstehen. Bei vielen Tokens wird gemittelt.

### Was ein neuronales Netz und ein Transformer beitragen

Ein neuronales Netz verknüpft einfache Rechenschritte mit veränderbaren Gewichten. Eine lineare Schicht bildet etwa aus einer Liste von Zahlen neue gewichtete Summen. Eine **Aktivierungsfunktion** wie ReLU oder GELU fügt eine nichtlineare Veränderung hinzu. Ohne solche Veränderungen würden viele hintereinandergelegte lineare Schichten insgesamt wieder nur eine lineare Abbildung ergeben.

Ein **Transformer** ist eine bestimmte Anordnung solcher Bausteine. **Attention** lässt eine Textposition Informationen anderer erlaubter Positionen unterschiedlich stark berücksichtigen. In «Lea stellte die Tasse ab, weil sie heiss war» kann die frühere Erwähnung der Tasse für eine Fortsetzung nützlich sein. Welche Verbindungen ein winziges Modell tatsächlich lernt, musst du beobachten; die Architektur allein garantiert kein Sprachverständnis.

Du baust einen **Decoder-only Transformer**: Token-Darstellungen laufen durch wiederholte Blöcke mit kausaler Self-Attention und kleinen neuronalen Netzen. Es gibt keinen zusätzlichen Encoder für eine zweite Eingabe. Das ursprüngliche Transformer-Paper beschreibt eine grössere Encoder-Decoder-Architektur; für MintCLM verwendest du nur die passende Grundidee. Vertiefung für später: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

### Der gesamte Ablauf

```text
Texte auswählen → bereinigen → Duplikate behandeln → Dokumente aufteilen
                                                     │
                           ┌─────────────────────────┴───────────────┐
                           ↓                                         ↓
                      Trainingstexte                           Validierungstexte
                           ↓                                         │
              Tokenizer festlegen oder lernen                       │
                           └──────── derselbe Tokenizer ─────────────┘
                                             ↓
                                         Token-IDs
                                             ↓
                              Eingaben und nächste Ziel-Tokens
                                             ↓
                     Modell → Vorhersagen → Loss → Gradienten → Update

Nach dem Training:
Starttext → Tokenizer → Modell → nächstes Token wählen → anhängen → wiederholen
```

Bei der Validierung fehlen Gradientenberechnung und Update. Ein BPE-Tokenizer wird ausschliesslich auf dem Trainingsteil gelernt. Der Validierungsteil darf seine Regeln nicht mitbestimmen.

### Was du erwarten darfst

Ein winziges Modell kann zunächst Zeichenhäufigkeiten und typische Übergänge lernen. Mit ausreichend passenden Daten und einem funktionierenden Transformer können Wortstücke, Interpunktion und kurze stilähnliche Passagen entstehen. Es kann trotzdem Wiederholungen, Unsinn und erfundene Aussagen erzeugen.

Ein solches Modell ist noch kein zuverlässiger Chat-Assistent und kein Faktenlexikon. Es bekommt durch mehr Parameter allein auch keine verlässliche Mathematik oder Recherchefähigkeit. **Dein überprüfbarer Erfolg ist zunächst: Der Ablauf funktioniert, der Loss entwickelt sich nachvollziehbar und du kannst erklären, warum.** Ein kleiner Versuch, dessen Grenzen du verstehst, kann der sinnvolle Endpunkt deiner ersten Projektetappe sein.

## Phase 0 Deine Arbeitsumgebung einrichten

**Mindestziel:** Ein Python-Skript und eine PyTorch-Rechnung laufen auf der CPU. GPU-Unterstützung prüfst du anschliessend. **Später möglich:** Arbeitsumgebungen für Workstation und DGX Spark. **Vorwissen:** Dateien und Ordner öffnen sowie Befehle in einer Konsole ausführen. **Ablage:** Einrichtung und Versionsangaben gehören in `docs/training/setup.md` innerhalb deines Repositorys.

### Deine tatsächliche Ausgangslage

Der geprüfte Projektordner ist:

```text
C:\Users\wunde\MintCLM\MintCLM 1.0
```

README, Anforderungen und Git-Ausschlüsse sind vorhanden. Die Python-Dateien und Konfigurationen sind derzeit Platzhalter. Es gibt noch kein startbares Trainingsprogramm. Auch eine Paketkonfiguration wie `pyproject.toml` ist noch nicht vorhanden.

In der aktuellen PowerShell war weder `python` noch `py` erreichbar. Das beweist nicht, dass auf dem ganzen Computer nirgends Python installiert ist; möglicherweise fehlt nur die Einbindung in diese Konsole. `nvidia-smi` erkannte eine NVIDIA GeForce RTX 5060 mit **8151 MiB angezeigtem Grafikspeicher**, also ungefähr 8 GB, und Treiber **621.24**. Das ist eine Bestandsaufnahme und noch kein erfolgreicher PyTorch-Test.

Der sichtbare Projektpfad liegt nicht unter dem gewöhnlichen kDrive-Ordner. Ob du ihn trotzdem mit kDrive synchronisierst, lässt sich am Pfad allein nicht erkennen. Prüfe die ausgewählten Synchronisierungsordner in kDrive.

### 0A Python und Treiber prüfen

Öffne PowerShell. Befehle aus `powershell`-Blöcken gehören dorthin. Python-Code gehört in eine `.py`-Datei oder in die Python-Konsole; kopiere diese beiden Arten nicht durcheinander.

```powershell
Get-Command python,py -ErrorAction SilentlyContinue
python --version
py -0p
nvidia-smi
```

Es ist in Ordnung, wenn einer der beiden Python-Befehle fehlt. Wichtig ist, einen funktionierenden Interpreter zu finden. Ein **Interpreter** ist das Programm, das deine Python-Anweisungen ausführt.

Wenn Python fehlt, verwende den offiziellen [Python Install Manager für Windows](https://docs.python.org/3/using/windows.html), erhältlich über [python.org](https://www.python.org/downloads/). Öffne nach der Installation eine neue Konsole. Für die unten geprüfte Paketkombination ist **normales 64-Bit-Python 3.12** eine konkrete Wahl; das ist keine Behauptung, dass es die neueste Python-Version sei.

Mit dem neuen Install Manager:

```powershell
py install 3.12
py -3.12 --version
py -3.12 -c "import platform, struct; print(platform.machine(), struct.calcsize('P') * 8)"
```

Für den Windows-Laptop erwarten wir `AMD64` und `64`. `AMD64` bezeichnet hier die x86-64-Architektur und sagt nichts darüber aus, ob dein Prozessor von Intel oder AMD stammt. Falls `py install` nicht verstanden wird, hast du möglicherweise den alten Launcher: `py -0p` zeigt dessen vorhandene Versionen; folge für die Installation dem verlinkten offiziellen Windows-Weg.

Wenn `nvidia-smi` scheitert, prüfe im Geräte-Manager, ob die NVIDIA-GPU erscheint, und verwende den für dein Notebook vorgesehenen Treiber von Lenovo oder NVIDIA. Die dort angezeigte CUDA-Version beschreibt die vom Treiber unterstützte CUDA-Version; sie ist weder die installierte PyTorch-Version noch ein Nachweis eines separat installierten CUDA Toolkits.

### 0B Quellcode und grosse Arbeitsdateien trennen

Vorschlag für einen zusätzlichen **lokalen, nicht synchronisierten** Arbeitsordner:

```text
C:\Users\wunde\MintCLM-local\
├── envs\mintclm\
├── data\raw\
├── data\processed\
├── data\tokenized\
├── tokenizers\
├── checkpoints\
├── logs\
└── outputs\
```

Prüfe, dass dieser Ordner nicht zu deinen kDrive-Synchronisierungsaufträgen gehört. Im Repository bleiben Code, kleine Einstellungen, Tests, Journal und kleine Ergebnisübersichten. Grosse Dateien liegen im lokalen Arbeitsordner. Die vorhandenen `data/`- und `artifacts/`-Unterordner dürfen vorerst Platzhalter bleiben.

**`.gitignore` beeinflusst nur Git. kDrive beachtet diese Datei nicht automatisch.** Ebenso ist Synchronisierung kein vollständiger Ersatz für ein Backup: Bewahre später wichtige fertige Checkpoints zusätzlich an einem bewusst gewählten zweiten Ort auf. Kopiere abgeschlossene Dateien; lasse nicht zwei Rechner denselben laufend geschriebenen Trainingsstand bearbeiten.

### 0C Eine virtuelle Umgebung erstellen

Eine virtuelle Umgebung ist ein eigener Satz von Python-Paketen für dieses Projekt. So kann MintCLM andere Paketversionen verwenden als ein anderes Projekt. Dein Quellcode liegt ausserhalb dieser Umgebung. Auf einem anderen Rechner erstellst du sie neu; du kopierst nicht den Windows-Umgebungsordner. Hintergrund: [Python-Dokumentation zu venv](https://docs.python.org/3/library/venv.html).

Nach erfolgreicher Python-Prüfung:

```powershell
Set-Location -LiteralPath 'C:\Users\wunde\MintCLM\MintCLM 1.0'
py -3.12 -m venv 'C:\Users\wunde\MintCLM-local\envs\mintclm'
& 'C:\Users\wunde\MintCLM-local\envs\mintclm\Scripts\Activate.ps1'
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
```

`sys.executable` muss auf die neue Umgebung zeigen. `python -m pip` verwendet den Paketinstaller genau dieses Python-Interpreters.

Falls PowerShell die Aktivierung blockiert, kannst du die Umgebung ohne Änderung der Ausführungsrichtlinie benutzen:

```powershell
& 'C:\Users\wunde\MintCLM-local\envs\mintclm\Scripts\python.exe' -m pip --version
```

Ersetze dann in den folgenden Befehlen das führende `python` durch `& 'C:\Users\wunde\MintCLM-local\envs\mintclm\Scripts\python.exe'`. Aktivierung ist eine Bequemlichkeit. Stelle auch im Editor diesen Interpreter ein. Mit `deactivate` verlässt du eine aktivierte Umgebung.

### 0D PyTorch passend installieren

Prüfe vor der Installation den [offiziellen PyTorch-Installationsassistenten](https://pytorch.org/get-started/locally/): **Stable, Windows, Pip, Python, passende CUDA-Variante**. Für die RTX-50-Generation brauchst du einen Build mit Blackwell-Unterstützung. PyTorch führte diese mit den CUDA-12.8-Builds von Version 2.7 ein; alte CUDA-Builds sind deshalb kein beliebiger Ersatz. Siehe [PyTorch 2.7 Release](https://pytorch.org/blog/pytorch-2-7/).

**Zum Stand dieser Anleitung offiziell gelistetes konkretes Beispiel:** PyTorch **2.14.1 mit CUDA 13.0** für Windows x86-64 und Python 3.12. Die Versionsseite nennt diesen CUDA-Index; im Paketverzeichnis ist das passende `cp312`-Paket für `win_amd64` vorhanden. Das ist eine verifizierte verfügbare Kombination, aber noch keine auf deinem Laptop erprobte Installation. [Offizielle Versionsbefehle](https://pytorch.org/get-started/previous-versions/) und [offizielles CUDA-13.0-Paketverzeichnis](https://download.pytorch.org/whl/cu130/torch/).

In der aktivierten Umgebung:

```powershell
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy pytest
```

`torchvision` und `torchaudio` aus manchen offiziellen Beispielen brauchst du für dieses Textprojekt zunächst nicht. Die benötigten CUDA-Laufzeitbibliotheken kommen beim üblichen PyTorch-Paket mit beziehungsweise werden als Paketabhängigkeiten installiert; ein separates CUDA Toolkit ist für die ersten Tensorrechnungen normalerweise nicht nötig. Für selbst kompilierte Erweiterungen kann sich das später ändern.

Wenn du die Anleitung später benutzt, prüfe das dann angebotene Stable-Paket erneut und dokumentiere deinen tatsächlichen Befehl. Wähle nicht nur nach einer hohen Versionsnummer: Betriebssystem, Prozessorarchitektur, Python, GPU-Unterstützung und Treiber müssen zusammenpassen.

### 0E Die GPU wirklich rechnen lassen

Lege später selbst `notebooks/check_setup.py` im Projekt an und führe dieses kurze Diagnosebeispiel aus:

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA-Build:", torch.version.cuda)
print("CUDA verfügbar:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(8, 8, device=device, requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
if device.type == "cuda":
    torch.cuda.synchronize()
print("Rechnung auf:", x.device)
print("Gradienten endlich:", torch.isfinite(x.grad).all().item())
```

`@` ist Matrixmultiplikation. `backward()` berechnet Gradienten. Du musst diese Schritte heute noch nicht vollständig erklären; hier prüfen sie mehr als nur den Import. Das Programm sollte ohne Fehlermeldung enden und `Gradienten endlich: True` ausgeben. Bei erfolgreicher GPU-Nutzung steht beim Gerät `cuda:0`.

```powershell
python 'C:\Users\wunde\MintCLM\MintCLM 1.0\notebooks\check_setup.py'
```

Typische Befunde:

| Befund | Nächster sinnvoller Schritt |
|---|---|
| `No module named torch` | Mit `sys.executable` und `python -m pip --version` prüfen, ob Installation und Ausführung dieselbe Umgebung nutzen. |
| `torch.version.cuda` ist `None` | Du verwendest einen CPU-Build. Den passenden GPU-Build nach offizieller Anleitung in einer passenden Umgebung installieren. |
| CUDA-Build vorhanden, aber `is_available()` ist falsch | Treiber, sichtbare GPU und gegebenenfalls Container-GPU-Zugriff prüfen. |
| `no kernel image` oder Architekturwarnung | Prüfen, ob der installierte PyTorch-Build die GPU-Generation unterstützt. Nicht einfach das Modell ändern. |
| Speicherfehler | Erst Batch und Kontext verkleinern; mit der winzigen Diagnose beginnen. |

**Wenn CUDA Zeit kostet:** Mache die Lernübungen auf der CPU weiter. Falls schon der Import der GPU-Version scheitert, erstelle eine getrennte Umgebung `mintclm-cpu` und installiere dort die offiziell gelistete CPU-Variante:

```powershell
py -3.12 -m venv 'C:\Users\wunde\MintCLM-local\envs\mintclm-cpu'
& 'C:\Users\wunde\MintCLM-local\envs\mintclm-cpu\Scripts\Activate.ps1'
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pytest
```

Wenn die Aktivierung blockiert wird, verwende wie in Phase 0C den vollständigen Interpreterpfad, diesmal mit `mintclm-cpu`. Für Zeichen-IDs, Gradientenübungen und ein sehr kleines Modell reicht die CPU als Einstieg.

### 0F Deine vorhandene requirements.txt verwenden

Die Datei enthält derzeit `torch>=2.7`, `numpy>=1.26`, `PyYAML>=6.0`, `tqdm>=4.66`, `wandb>=0.18`, `gradio>=5.0` und `pytest>=8.0`.

| Paket | Deine Nutzung |
|---|---|
| torch | Tensoren, automatische Ableitungen, einfache Schichten und AdamW; deine Modellarchitektur entwickelst du selbst. |
| numpy | Arrays, Datenstatistik und später kompakte Datenspeicherung. |
| pytest | Automatisches Ausführen deiner Bausteintests. |
| PyYAML | Später Einstellungen aus YAML-Dateien lesen. |
| tqdm | Optionale Fortschrittsanzeige. |
| wandb | Optionaler Dienst für Experimentprotokolle; zunächst reichen lokale Dateien. |
| gradio | Optionale spätere Oberfläche. |

Für den Einstieg reichen die bereits genannten Minimalpakete. Wenn du die vollständige Liste nutzen möchtest, installiere **zuerst den passenden PyTorch-Build**, dann:

```powershell
python -m pip install -r 'C:\Users\wunde\MintCLM\MintCLM 1.0\requirements.txt'
python -m pip check
```

`torch>=2.7` ist nur eine Untergrenze. Sie legt weder eine feste Version noch den CUDA-Build fest. Prüfe nach Paketänderungen nochmals Version und GPU-Rechnung. Für einen dokumentierten Trainingslauf sichere eine Momentaufnahme:

```powershell
python -m pip freeze | Out-File -Encoding utf8 'C:\Users\wunde\MintCLM\MintCLM 1.0\docs\training\packages-windows.txt'
```

Notiere zusätzlich Python-Version, Plattform, Installationsbefehl und Treiber. `pip freeze` allein beschreibt nicht die ganze GPU-Umgebung. Kopiere eine Windows-Paketliste nicht blind auf ARM-Linux.

### 0G Auf Linux und auf den DGX Spark wechseln

Unter Linux sieht eine lokale Umgebung beispielsweise so aus:

```bash
python3 --version
uname -m
python3 -m venv "$HOME/mintclm-local/envs/mintclm"
source "$HOME/mintclm-local/envs/mintclm/bin/activate"
```

Falls `venv` fehlt, verwende das zu deiner Linux-Distribution passende Paket oder die bereitgestellte Umgebung des Betreuers. Die PyTorch-Installation wählst du danach für **Linux und die angezeigte Architektur**. Windows, WSL und Linux sind getrennte Umgebungen; eine in Windows funktionierende Installation wird nicht automatisch in WSL verfügbar.

Der DGX Spark hat einen Arm-Prozessor und gemeinsamen Speicher für CPU und GPU. Für ihn sind ARM64-kompatible Pakete beziehungsweise ein passender NVIDIA-PyTorch-Container nötig. Ein Container ist eine vorbereitete Softwareumgebung; dein eigener Modellcode bleibt darin derselbe. Kläre mit der betreuenden Person einen aktuell unterstützten Container samt Versionskennung. Daten und Checkpoints müssen über eingebundene Ordner ausserhalb eines kurzlebigen Containers erhalten bleiben. [DGX-Spark-Hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [Softwareübersicht](https://docs.nvidia.com/dgx/dgx-spark/software.html) und [NGC-Umgebungen](https://docs.nvidia.com/dgx/dgx-spark/ngc.html).

Übertrage Code, Einstellungen und benötigte Daten, erstelle die Umgebung neu und führe dort dieselben kleinen Tests aus. Verwende im Python-Code `pathlib.Path` statt überall Windows-Pfade fest einzubauen.

**Weiter, wenn:** Du weisst, welcher Interpreter läuft, und mindestens eine CPU-Rechnung gelingt. Ein noch ungelöstes GPU-Problem muss die nächsten Lernphasen nicht blockieren.

## Phase 1 Die benötigten Grundlagen lernen

**Mindestziel:** Du kannst Formen von Arrays erklären, einen Gradienten berechnen lassen und eine einfache Lernschleife beschreiben. **Später möglich:** Lineare Algebra, Statistik und Optimierung vertiefen. **Vorwissen:** Deine bisherigen Python-Grundlagen und das Ausführen eines kleinen Skripts. **Ablage:** Kleine Versuche in `notebooks/`, Erkenntnisse im Projektjournal. Lerne jeweils das, was du im nächsten Baustein brauchst.

### Dein Lernpfad mit überprüfbaren Übungen

| Thema | Wofür du es brauchst | Kleine Übung | Erreicht, wenn |
|---|---|---|---|
| Funktionen und Rückgabewerte | Verarbeitungsschritte getrennt testen | Schreibe `count_chars(text)`, die Zeichen zählt. | Zwei kurze Testtexte ergeben die erwarteten Zahlen. |
| Listen, Dictionaries, Schleifen | Token-IDs und BPE-Paare verwalten | Erzeuge eine Zeichen-zu-ID-Tabelle. | Du kannst ein Zeichen nachschlagen und die ID zurückübersetzen. |
| Dateien, UTF-8, JSON und pathlib | Korpus und Einstellungen speichern | Lies einen Text mit `ä` ein und speichere ein kleines Dictionary als JSON. | Inhalt und Sonderzeichen bleiben erhalten. |
| Klassen und Module | Modell und Tokenizer strukturieren | Eine kleine Klasse speichert eine Tabelle und hat eine `encode`-Methode. | Du kannst erklären, was `self` und Objektzustand sind. |
| NumPy und Formen | Daten prüfen und kompakt ablegen | Erzeuge sechs Zahlen, forme sie zu `(2, 3)` um, berechne Zeilenmittelwerte. | Du kannst die Formen vor dem Ausführen vorhersagen. |
| PyTorch-Tensoren | Die Modellrechnung ausdrücken | Multipliziere `(2, 3)` mit `(3, 4)`. | Du erklärst das Ergebnis `(2, 4)` und den Unterschied zwischen `*` und `@`. |
| Autograd | Gradienten berechnen | Verwende das Beispiel unten mit `w = 2`. | Du verstehst, warum `w.grad` gleich 4 ist. |
| Neuronales Netz | Lernbare Rechenschritte verbinden | Lerne mit `nn.Linear(1, 1)` die Beziehung `y = 2*x + 1`. | Der mittlere quadratische Fehler sinkt, auch bei einem neuen Testwert. |
| Loss und Optimierer | Parameter sinnvoll ändern | Lass einen einzelnen Parameter in Richtung 5 lernen. | Du kannst Vorhersage, Loss, Gradient und Update unterscheiden. |
| Batches und Validierung | Lernen und Prüfen trennen | Halte einige einfache Zahlenpaare vom Training zurück. | Du misst den Fehler darauf ohne Parameteränderung. |

Nutze passend dazu das [offizielle Python-Tutorial](https://docs.python.org/3.12/tutorial/), den [NumPy-Einstieg](https://numpy.org/doc/stable/user/absolute_beginners.html) und [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). Die PyTorch-Beispiele verwenden teilweise Bilder: Du kannst die Tensor- und Lernideen trotzdem verwenden, ohne ein Bildmodell nachzubauen.

### Arrays und Tensorformen greifbar machen

```python
import numpy as np

a = np.arange(6).reshape(2, 3)
print(a)
print(a.shape)
print(a.mean(axis=1))
```

Du erhältst zwei Zeilen mit je drei Zahlen: `[0, 1, 2]` und `[3, 4, 5]`. `shape` ist `(2, 3)`. `axis=1` fasst die Spalten innerhalb jeder Zeile zusammen; die Mittelwerte sind `[1., 4.]`. `reshape` ändert die Anordnung, nicht die Anzahl der Elemente.

Ein Tensor ist hier ein Zahlenfeld mit beliebig vielen Achsen. PyTorch-Tensoren können zusätzlich auf einer GPU liegen und an automatischen Ableitungen beteiligt sein. Für Token-IDs verwenden wir `torch.long`, für lernbare Gewichte zunächst Gleitkommazahlen wie `float32`. `shape`, `dtype` und `device` beantworten drei unterschiedliche Fragen: **Welche Form? Welche Art Zahlen? Auf welchem Gerät?** [PyTorch-Tensoren](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).

### Einen Gradienten selbst nachvollziehen

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
loss = w ** 2
loss.backward()
print(loss.item())
print(w.grad.item())
```

Beide Ausgaben sind `4.0`, aus unterschiedlichen Gründen: Der Loss ist `2²`; die Steigung von `w²` bei 2 ist `2*w = 4`. Ein positiver Gradient bedeutet hier: Ein kleines Erhöhen von `w` erhöht den Loss. Zum Verkleinern gehen wir ein Stück in die Gegenrichtung. Autograd verfolgt die Rechenschritte und berechnet diese Ableitungen für dich. Vertiefung: [PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).

Als nächste Übung ersetzt du den Loss durch `(w - 5) ** 2`. Bei `w = 2` ist der Gradient `-6`. Erkläre vor dem Ausführen, weshalb ein Update gegen diesen Gradienten `w` erhöhen sollte.

**Typische Fehler:** Eine Ganzzahl kann nicht als normaler lernbarer Gleitkommaparameter dienen; Gradienten addieren sich bei mehreren `backward()`-Aufrufen; `detach()` oder eine Umwandlung in normale Python-Zahlen kann die Ableitungskette unterbrechen. Wenn eine Form unerwartet ist, drucke nach jedem Schritt `shape` aus. Versuche nicht, Formprobleme nur durch zufälliges `reshape` zu beseitigen.

**Weiter, wenn:** Du kannst ein `(B, T)`-Array als `B` Beispiele mit je `T` Tokens lesen und erklären, warum der Optimierer Gradienten braucht.

## Phase 2 Kleine Textdaten auswählen und aufteilen

**Ziel:** Du weisst, was deine Daten enthalten und was später als unbekannter Text geprüft wird. **Vorwissen:** Textdateien, Listen und Dictionaries. **Ablage:** `src/mintclm/data/cleaning.py`, `deduplication.py`, `split.py`; Quellen in `docs/data/sources.md`, Verarbeitung in `docs/data/preparation.md`. Die grossen Originaldateien liegen im lokalen Arbeitsordner.

**Mindestziel:** Einige selbst geschriebene kurze Texte. Für den ersten technischen Test genügen wenige Sätze; später kannst du beispielsweise einige Hundert Kilobyte bis wenige Megabyte sorgfältig ausgewählter Texte erproben. Das ist eine Arbeitsgrössen-Empfehlung, keine Qualitätsgarantie. **Später möglich:** Mehrere Quellen, Sprachfilter und Erkennung sehr ähnlicher Texte.

1. Schreibe zunächst einige eigene Absätze. So kennst du den Inhalt und kannst Veränderungen leicht beurteilen.
2. Erfasse je Dokument eine Kennung, Herkunft, Sprache und die für deine Nutzung geklärten Rechte. Verwende für den Anfang keine privaten Chats oder personenbezogenen Sammlungen.
3. Lies die Texte als UTF-8. Zähle Dokumente, Zeichen und Bytes; ein Umlaut oder Emoji kann mehr als ein Byte belegen.
4. Schaue Anfang, Ende und zufällige Abschnitte an. Suche leere Texte, defekte Zeichen, wiederholte Kopfzeilen und unpassende Inhalte.
5. Definiere eine kleine Bereinigung, etwa einheitliche Zeilenenden. Bewahre Rohdaten unverändert auf. Entferne Leerzeichen, Satzzeichen oder Grossschreibung nicht ohne konkrete Begründung.
6. Entferne identische bereinigte Dokumente. Mit einem Hash, beispielsweise SHA-256, bekommst du eine kompakte Kennung des Inhalts. Behandle auch erkennbare Varianten derselben Vorlage gemeinsam.
7. Teile **Dokumente oder zusammengehörige Dokumentgruppen** vor der Tokenisierung in Training und Validierung auf. Eine anfängliche Wahl sind etwa 90/10 Prozent bei ausreichend vielen Dokumenten. Notiere einen festen Zufalls-Seed und speichere die tatsächlichen Dokument-IDs je Teil.

Ein **Seed** ist ein Startwert für einen Zufallszahlengenerator. Dieselbe Auswahl lässt sich damit unter denselben Bedingungen leichter wiederholen. Für Reproduzierbarkeit ist das gespeicherte Aufteilungsverzeichnis noch hilfreicher als nur die Seed-Zahl.

**Warum vor den Trainingsfenstern aufteilen?** Zwei stark überlappende Fenster desselben Textes dürfen nicht in verschiedenen Teilen landen. Sonst sieht die Validierung unbekannt aus, enthält aber beinahe bereits gelernte Abschnitte. Das heisst Datenleck oder *data leakage*.

Bei nur einem langen Text kannst du einen zusammenhängenden letzten Abschnitt zurückhalten und erst danach innerhalb beider Teile Fenster bilden. Es darf kein Fenster die Grenze überschreiten. Notiere, dass beide Teile aus derselben Quelle stammen. Für eine spätere Abschlussbewertung kannst du zusätzlich einen festen Testteil vorsehen; er bleibt bei Modellentscheidungen unangetastet. Bei winzigen Spielzeugdaten wären drei Teile zunächst eher verwirrend als hilfreich.

**Prüfung:** Dokumentzahlen und Längen vor und nach der Bereinigung protokollieren; einige Beispiele nebeneinander ansehen; sicherstellen, dass kein Inhalts-Hash in Training und Validierung vorkommt. Das erkennt exakte, aber noch keine leicht veränderten Duplikate.

**Fehler eingrenzen:** Wenn fast nichts übrig bleibt, prüfe deine Filter an fünf Dokumenten. Wenn Validierung später erstaunlich gut wirkt, prüfe zuerst gemeinsame Texte und Vorlagen. Wenn eine Datei unlesbar wird, prüfe die Zeichenkodierung vor Änderungen am Modell.

**Weiter, wenn:** Du kannst Herkunft, Bereinigung und Aufteilung deiner kleinen Sammlung erklären und die Dokumentlisten erneut erzeugen oder laden.

## Phase 3 Eine einfache Tokenisierung bauen

**Ziel:** Text zuverlässig in Zahlen und zurück umwandeln. **Vorwissen:** Dictionaries, Indizes und UTF-8-Grundidee. **Ablage:** Ein neuer kleiner Baustein `src/mintclm/tokenizer/char.py` oder `byte.py`; `bpe.py` bleibt für später. Tests in `tests/test_tokenizer.py`.

**Mindestziel:** Ein Zeichen-Tokenizer für einen festgelegten Spielzeugtext. **Später möglich:** Byte-Tokenizer und BPE.

### Erst einmal ein Zeichen pro Token

```python
text = "MINT!"
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

ids = [stoi[ch] for ch in text]
restored = "".join(itos[i] for i in ids)
print(ids)
assert restored == text
```

`set` sammelt unterschiedliche Zeichen. `sorted` bringt sie in eine feste Reihenfolge. `enumerate` vergibt IDs ab 0. Hier ist die Tabelle `!→0, I→1, M→2, N→3, T→4`; die Ausgabe lautet `[2, 1, 3, 4, 0]`. `assert` bricht ab, wenn die folgende Bedingung falsch ist.

Formuliere daraus selbst zwei Funktionen: `encode(text)` und `decode(ids)`. Teste auch Leerzeichen und Zeilenumbrüche. Für dieses Minibeispiel baust du das Vokabular aus dem ganzen Spielzeugtext, weil du nur die Umwandlung untersuchst. **Für eine echte Auswertung** leitest du ein gelerntes Zeichenvokabular nur aus Trainingstexten ab oder legst das Alphabet im Voraus fest.

Ein unbekanntes Zeichen führt sonst zu einem `KeyError`. Du kannst eine klar dokumentierte Fehlermeldung ausgeben. Ein `<UNK>`-Token für unbekannte Zeichen wäre möglich, verhindert aber die exakte Wiederherstellung dieses ursprünglichen Zeichens.

### Bytes als unkomplizierte nächste Stufe

```python
text = "Grüezi 🙂"
ids = list(text.encode("utf-8"))
restored = bytes(ids).decode("utf-8")
assert restored == text
```

Ein Byte hat einen Wert von 0 bis 255. Mit diesen 256 Grundtokens kannst du jeden gültigen UTF-8-Text darstellen, ohne ein Zeichenalphabet aus der Validierung abzuleiten. Dafür werden manche Zeichen mehrere Tokens lang. Ein Byte-Tokenizer lernt noch keine Zusammenführungen; genau das ergänzt später BPE.

Die verlustfreie Prüfung gilt für den **Text, den du an den Tokenizer übergibst**. Wenn die Bereinigung davor Leerzeichen verändert hat, kann der Tokenizer die entfernten Originalzeichen nicht zurückholen.

**Prüfung:** Leerer Text, normale Wörter, doppelte Leerzeichen, `\n`, Umlaute, Emoji und wiederholte Zeichen müssen für die jeweils unterstützte Eingabe korrekt zurückkommen. Speichere und lade die Zeichentabelle, statt die IDs bei jeder Nutzung neu zu vergeben.

**Typische Fehler:** `str.split()` wirft Informationen über Leerzeichen weg; unterschiedliche Tabellen zwischen Training und Generierung machen IDs bedeutungslos; einzelne UTF-8-Bytes lassen sich nicht immer getrennt als Zeichen decodieren.

**Weiter, wenn:** `decode(encode(text)) == text` für deine Testfälle gilt und du erklären kannst, warum ID 100 keine «grössere Bedeutung» als ID 10 hat.

## Phase 4 Die richtigen Trainingsbeispiele erzeugen

**Ziel:** Zu jedem sichtbaren Token passt genau das folgende Zieltoken. **Vorwissen:** Slicing und Tensorformen. **Ablage:** `src/mintclm/data/dataset.py`, `scripts/tokenize_data.py`, `tests/test_data.py`.

**Mindestziel:** Kurze Fenster fester Länge in einem Tensor im Arbeitsspeicher. Kein Padding, kein Memmap und keine parallelen Ladeprozesse nötig. **Später möglich:** Variable Längen, mehrere Leseprozesse oder kompakte Dateien.

```python
import torch

tokens = torch.tensor([2, 1, 3, 4, 0], dtype=torch.long)
x = tokens[:-1]
y = tokens[1:]
print(x.tolist())  # [2, 1, 3, 4]
print(y.tolist())  # [1, 3, 4, 0]
```

Das entspricht `MINT` als Eingabe und `INT!` als Ziel. Die Ausgabe an der Position des `M` soll also das `I` vorhersagen. Das Verschieben passiert in deinem Ablauf **genau einmal**. Wenn du bereits `x` und `y` so baust, darf die Loss-Berechnung sie nicht nochmals verschieben.

Für Kontextlänge `T` brauchst du `T+1` aufeinanderfolgende Tokens. Ein Fenster ab Start `s` ergibt:

```text
x = tokens[s : s+T]
y = tokens[s+1 : s+T+1]
```

Bei einem Strom mit `N` Tokens gilt `0 ≤ s ≤ N-T-1`. `torch.randint(0, N-T, ...)` verwendet eine ausgeschlossene obere Grenze und passt dazu, sofern `N > T`.

Mehrere Fenster bilden einen Batch. Unsere Notation für alle folgenden Kapitel:

| Zeichen | Bedeutung | Erstes Beispiel |
|---|---|---|
| B | Batchgrösse | 4 |
| T | Kontextlänge | 32 |
| C | Breite der internen Darstellung | 64 |
| V | Vokabulargrösse | 256 bei Bytes ohne Sondertokens |
| H | Anzahl Attention-Köpfe | 4 |

`x` und `y` haben Form `(B, T)` und Datentyp `torch.long`. Das Modell liefert später `(B, T, V)`: für jedes Beispiel und jede Position einen Wert pro möglichem nächsten Token.

Beginne mit Fenstern innerhalb einzelner ausreichend langer Dokumente. Kurze Dokumente kannst du für diesen ersten Versuch zurückstellen und diese Auswahl dokumentieren. Wenn du später Dokumente zu einem Strom zusammensetzt, füge ein eigenes End-of-document-Token ein und erkläre die gewählte Strategie. Ein solches Token markiert eine Grenze, **sperrt aber nicht automatisch Attention über diese Grenze**. Wirklich unabhängige gepackte Dokumente benötigen zusätzlich eine passende Attention-Maske.

**Prüfung:** Drucke Eingabe und Ziel sowohl als IDs als auch als Text aus. Prüfe `x.shape == y.shape`, ganzzahlige IDs und `0 <= ID < V`. Training und Validierung müssen getrennte Datenobjekte verwenden. Beim ersten `DataLoader` ist `num_workers=0` unter Windows eine einfache Wahl; mehrere Worker und der nötige `if __name__ == "__main__":`-Einstieg kommen später.

**Typische Fehler:** Ziel entspricht der Eingabe, letztes Fenster ist zu kurz, ein zufälliges Fenster überquert die Datengrenze oder Padding wird als echter Inhalt bewertet. Der erste Fehler kann einen erfreulich kleinen Loss liefern, obwohl das Modell nur abschreibt.

**Weiter, wenn:** Du kannst fünf zufällige Fenster von Hand prüfen und weisst für jede Position, welches Token vorhergesagt werden soll.

## Phase 5 Ein winziges erstes Sprachmodell bauen

**Ziel:** Die ganze Lernaufgabe verstehen, bevor Attention hinzukommt. **Vorwissen:** Parameter, Tensorformen und verschobene Ziele. **Ablage:** Ein neuer Baustein `src/mintclm/model/bigram.py`; kleine Versuche in `notebooks/`, Tests in `tests/test_model.py` und `tests/test_training.py`.

**Mindestziel:** Ein Bigram-Modell. **Später möglich:** Es als Vergleich für deinen Transformer behalten. *Bigram* bedeutet hier: Das Modell sagt das nächste Token ausschliesslich anhand des unmittelbar vorherigen Tokens voraus. Es ist bereits ein kausales Sprachmodell, verwendet aber noch keinen Transformer.

Stell dir eine Tabelle mit `V` Zeilen und `V` Spalten vor. Jede Zeile gehört zu einem gerade gelesenen Token, jede Spalte zu einem möglichen Folgetoken. Die Tabelleneinträge sind lernbare Zahlen. Für `a` liest du die Zeile `a` aus und erhältst die Bewertungen für alle möglichen nächsten Tokens.

Diese rohen Bewertungen heissen **Logits**. Mit **Softmax** lassen sie sich in positive Wahrscheinlichkeiten umwandeln, die zusammen 1 ergeben. Für die Loss-Berechnung gibst du die rohen Logits direkt an PyTorch weiter.

Deine Umsetzungsschritte:

1. Definiere eine Klasse, die von `torch.nn.Module` erbt. Damit werden ihre lernbaren Teile von PyTorch erfasst.
2. Lege eine lernbare Tabelle der Form `(V, V)` an. `nn.Embedding(V, V)` ist dafür ein einfacher PyTorch-Grundbaustein: Er wählt Zeilen anhand von IDs aus.
3. Implementiere `forward(idx)` so, dass zu `(B, T)` Eingabe die Form `(B, T, V)` zurückkommt.
4. Drucke die Parameterzahl aus: `sum(p.numel() for p in model.parameters())`. Für diese Tabelle allein sind es `V * V` Parameter.
5. Verbinde das Modell mit der Trainingsschleife aus Phase 7. Verwende zuerst ein eindeutiges Spielzeugmuster wie `abcabcabc...`.

Lies dafür direkt Phase 7A bis 7C und kehre nach dem ersten gelungenen Bigram-Lernversuch zu Phase 6 zurück. Du brauchst den Transformer für diesen ersten Versuch noch nicht.

Warum dieses Muster? Nach `a` folgt immer `b`, nach `b` immer `c`, nach `c` immer `a`. Ein Bigram-Modell kann das lernen. Bei einem echten Satz kann auf dasselbe Zeichen vieles folgen; dann kann ein Bigram-Modell selbst durch Auswendiglernen nicht jeden Zieltoken sicher treffen.

**Prüfung:** Für zwei Eingabebeispiele mit je vier Tokens muss die Ausgabe `(2, 4, V)` haben. Ändere die Token-ID an genau einer Position: Beim Bigram-Modell darf sich nur die Ausgabe dieser Position ändern. Nach dem Spielzeugtraining sollte eine Generierung typische Übergänge des Musters zeigen.

**Typische Fehler:** Eine normale, nicht registrierte Tabelle wird vom Optimierer übersehen; Eingabe-IDs liegen ausserhalb des Vokabulars; du erwartest lange Zusammenhänge von einem Modell, das nur ein Token berücksichtigt. Prüfe zuerst die Parameterliste und die Ausgabeform.

**Weiter, wenn:** Du kannst erklären, welche Tabellenzeile eine Vorhersage erzeugt und weshalb dieses Modell «Heute scheint die» nicht als ganzen Zusammenhang berücksichtigt.

## Phase 6 Deinen eigenen kleinen Transformer entwickeln

**Ziel:** Vorhersagen können mehrere frühere Tokens berücksichtigen. **Vorwissen:** Matrixmultiplikation, Embeddings, Softmax, Formen und die Trainingsaufgabe. **Ablage:** `src/mintclm/model/config.py`, `attention.py`, `layers.py`, `transformer.py`; Entwurfsnotizen in `docs/architecture/design.md`.

**Mindestziel:** Ein kleiner Decoder mit selbst aufgebauter kausaler Attention und einem oder zwei Blöcken. **Später möglich:** Mehr Blöcke, andere Normierung, RoPE, GQA und SwiGLU. Die dafür vorhandenen Platzhalter müssen jetzt noch keinen Code enthalten.

### 6A Die erste Architektur festlegen

Eine mögliche kleine Ausgangskonfiguration ist:

| Einstellung | Vorschlag für einen Lernversuch |
|---|---|
| Tokens | Zeichen oder die 256 Bytes; zusätzliche Sondertokens bei `V` mitrechnen |
| Kontextlänge `T` | Zunächst 32, danach beispielsweise 64 |
| Modellbreite `C` | 64 |
| Attention-Köpfe `H` | 4, also 16 Zahlen je Kopf |
| Blöcke | Zunächst 1, danach 2 |
| Breite im Feedforward-Netz | `4 * C`, hier 256 |
| Batchgrösse `B` | Zunächst 2 oder 4 |
| Dropout | Für die ersten Korrektheitsprüfungen 0 |
| Rechengenauigkeit | Zunächst float32 |

Das sind überschaubare Startwerte, keine gemessene optimale Konfiguration für deine GPU. Zähle die tatsächlichen Parameter im Code und miss den Speicherbedarf.

Deine erste Architektur kann so aussehen:

```text
Token-IDs (B,T)
    ↓ Token-Embedding + gelerntes Positions-Embedding
Darstellungen (B,T,C)
    ↓ ein oder zwei Decoder-Blöcke
    ↓ abschliessende Normierung
    ↓ lineare Ausgabe auf V mögliche Tokens
Logits (B,T,V)

Ein Decoder-Block:
h = h + Attention(Norm(h))
h = h + Feedforward(Norm(h))
```

Ein **Embedding** übersetzt eine ID in einen lernbaren Zahlenvektor. Die Positions-Embeddings ergänzen, an welcher Stelle ein Token steht. **Normierung** hält die Grössenordnungen der Darstellungen in einem kontrollierten Bereich. Die Addition des alten `h` heisst **Residualverbindung**: Der Block ergänzt eine vorhandene Darstellung, statt sie vollständig ersetzen zu müssen. Die hier gezeigte Anordnung mit Normierung vor dem jeweiligen Teil heisst *Pre-Norm*.

Du verwendest PyTorch für Grundbausteine wie `nn.Linear`, `nn.Embedding`, zunächst `nn.LayerNorm` und eine Aktivierungsfunktion. Die Anordnung, Attention-Rechnung, kausale Maske, Datenführung und Trainingslogik entwickelst du selbst. Ein kompletter `nn.Transformer` oder ein vortrainiertes Modell würde diesen Lernschritt vorwegnehmen. Wenn du später auch LayerNorm selbst umsetzt, vergleiche zunächst die Ergebnisse mit dem einfachen PyTorch-Baustein.

### 6B Attention in kleinen Schritten verstehen

Aus jeder Darstellung berechnest du drei neue Darstellungen durch lernbare lineare Abbildungen:

- **Query Q:** Welche Information suche ich an dieser Position?
- **Key K:** Woran könnte eine andere Position als passend erkannt werden?
- **Value V:** Welche Information liefert diese Position bei einer Verbindung?

Das `V` in *Value* ist hier eine Matrix; in unseren Formtabellen bezeichnet `V` dagegen die Vokabulargrösse. Im Code sind Namen wie `value` und `vocab_size` eindeutiger.

Für einen Kopf vergleichst du jede Query mit den erlaubten Keys. Ein Skalarprodukt ist dabei eine Summe paarweise multiplizierter Zahlen. Ähnliche Ausrichtung kann einen hohen Vergleichswert ergeben. Durch Softmax werden daraus Gewichte; eine gewichtete Summe der Values liefert die neue Darstellung. Die vereinfachte Formel lautet:

```text
scores  = (Q @ Kᵀ) / sqrt(d)
weights = softmax(maskiere(scores))
output  = weights @ value
```

`d` ist die Breite eines Kopfes. Die Teilung durch `sqrt(d)` hält die Skala der Vergleichswerte handhabbar. `Kᵀ` vertauscht die letzten beiden Achsen. Du brauchst die Formel zunächst lesen und mit kleinen Formen ausprobieren zu können; ein vollständiger mathematischer Beweis ist kein Einstiegshindernis.

### 6C Die kausale Maske ausprobieren

Dieses **eigenständig ausführbare Rechenbeispiel** zeigt einen Kopf ohne Training:

```python
import math
import torch

torch.manual_seed(7)
T, d = 4, 2
q = torch.randn(T, d)
k = torch.randn(T, d)
value = torch.randn(T, d)

scores = (q @ k.T) / math.sqrt(d)
allowed = torch.ones(T, T, dtype=torch.bool).tril()
scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(scores, dim=-1)
output = weights @ value

print(weights)
print(output.shape)
assert torch.all(weights.triu(diagonal=1) == 0)
assert torch.allclose(weights.sum(dim=-1), torch.ones(T))
```

Die erlaubten Verbindungen sehen so aus:

```text
           Information von Position
           0  1  2  3
Position 0 ✓  ×  ×  ×
Position 1 ✓  ✓  ×  ×
Position 2 ✓  ✓  ✓  ×
Position 3 ✓  ✓  ✓  ✓
```

`tril()` behält das untere Dreieck einschliesslich der Diagonale. Eine Position darf ihr eigenes **Eingabe**token sehen, weil sie das **nächste** Token vorhersagen soll. Verbotene Werte werden vor Softmax zu minus unendlich; dadurch ist ihr Gewicht nach Softmax null. Die Diagonale bleibt erlaubt, damit keine Zeile vollständig gesperrt ist.

Baue danach Batch und Köpfe hinzu. Mit `C = H * d` werden aus `(B,T,C)` die Formen `(B,H,T,d)`. Die Scores erhalten `(B,H,T,T)`. Nach der Mischung setzt du die Köpfe wieder zu `(B,T,C)` zusammen und verwendest eine Ausgabeprojektion. **Multi-Head Attention** bedeutet mehrere solcher Kopf-Rechnungen mit unterschiedlichen lernbaren Abbildungen.

Schreibe nach jedem Schritt die erwartete Form auf. Ein `transpose` kann die Speicheranordnung verändern; zum erneuten Zusammenfassen ist `reshape` oft einfacher als ein unvorbereitetes `view`. Entscheidend ist aber, dass du die Achsen korrekt zurückordnest.

### 6D Den Block zusammensetzen

1. Token- und Positions-Embeddings addieren. Für Positionen reichen zunächst IDs `0` bis `T-1`.
2. Die selbst geschriebene Attention mit Residualverbindung ergänzen.
3. Ein Feedforward-Netz anfügen: `Linear(C, 4*C)`, GELU oder ReLU, dann `Linear(4*C, C)`. Es bearbeitet jede Position mit denselben Gewichten einzeln.
4. Normierung, zunächst `nn.LayerNorm(C)`, an den gezeigten Stellen verwenden. Über die letzte Achse `C` normieren, nicht über spätere Textpositionen.
5. Den vollständigen Block zunächst einmal anwenden.
6. Abschliessend normieren und mit `Linear(C, vocab_size)` auf Logits abbilden.
7. Alle Bausteine als registrierte Module speichern, etwa mit `nn.ModuleList` für mehrere Blöcke.

**Prüfung:** Ausgabeform, endliche Zahlen, kausales Verhalten und ein einziger vollständiger Vorwärts- und Rückwärtsschritt. Die konkrete Kausalitätsprüfung steht im Testkapitel. Erst danach mehrere Blöcke verwenden.

**Typische Fehler:** Softmax läuft über die falsche Achse; `C` ist nicht durch `H` teilbar; das obere Dreieck bleibt offen; Residualverbindungen addieren unterschiedliche Formen; die Maske liegt auf der CPU, die Gewichte auf der GPU. Bei `NaN` prüfst du, ob eine Attention-Zeile vollständig maskiert wurde.

**Weiter, wenn:** Du kannst den Weg von `(B,T)` nach `(B,T,V)` erklären und eine Änderung zukünftiger Eingabe-Tokens lässt frühere Ausgaben unverändert.

Später darfst du deine verständliche Attention durch eine optimierte PyTorch-Rechnung ersetzen. Vergleiche die Ergebnisse zuerst numerisch. Beachte die jeweilige Maskenbedeutung: Bei `scaled_dot_product_attention` bedeutet ein boolesches `True` eine erlaubte Position; andere APIs können andere Konventionen haben. Mit gleich langen Q/K-Sequenzen ist `is_causal=True` eine mögliche spätere Variante. [Offizielle SDPA-Dokumentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Phase 7 Den Trainingsprozess verstehen und durchführen

**Ziel:** Dein Modell verbessert seine Vorhersagen und du kannst das messen. **Vorwissen:** Loss, Gradienten, Batches und Modellaufruf. **Ablage:** `src/mintclm/training/pretrain.py`, später `scheduler.py`; Einstieg in `scripts/train.py`; Einstellungen in `configs/pretrain.yaml`; Tests in `tests/test_training.py`. Logs und Checkpoints liegen im lokalen Arbeitsordner.

**Mindestziel:** Erst einen festen kleinen Batch lernen, danach ein kurzes Training mit getrennter Validierung durchführen. **Später möglich:** Warmup, Cosine-Lernratenplan, Gradient Accumulation, Mixed Precision und längere Läufe.

### 7A Die Loss-Berechnung prüfen

Dein Modell liefert rohe Logits `(B,T,V)`, die Ziele haben `(B,T)`. Für Cross-Entropy kannst du alle Positionen zu einer Liste von Vorhersagen zusammenfassen:

```python
# Ausschnitt: model, x und y wurden vorher von dir erzeugt.
import torch.nn.functional as F

logits = model(x)
vocab_size = logits.shape[-1]
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    y.reshape(-1),
)
```

`-1` lässt PyTorch die passende zusammengefasste Grösse berechnen. Die Reihenfolge von Beispielen und Positionen muss in beiden Feldern übereinstimmen. Vor dieser Funktion verwendest du **kein zusätzliches Softmax**; Cross-Entropy verarbeitet Logits selbst. Die Ziele sind ganzzahlige Klassen-IDs. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

Wenn alle Tokens gleich wahrscheinlich wären, wäre der Loss `log(V)`. Bei 256 Tokens ist das ungefähr 5,545. Das ist ein grober Kontrollwert für eine gleichförmige Vorhersage, kein vorgeschriebener exakter Startwert deines zufällig initialisierten Modells.

### 7B Die erste Lernschleife zusammensetzen

Die Reihenfolge eines Updates ist:

1. Gradienten vom vorherigen Update zurücksetzen.
2. Vorhersagen und Loss berechnen.
3. Mit `backward()` Gradienten berechnen.
4. Bei Bedarf sehr grosse Gradienten begrenzen.
5. Den Optimierer die Parameter ändern lassen.

Dieses Muster setzt dein bereits gebautes `model` und einen Batch `x, y` voraus:

```python
import torch
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
model.train()

optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

print("Loss:", loss.item(), "Gradientennorm:", float(grad_norm))
```

`3e-4` bedeutet `0.0003` und ist nur ein möglicher Startwert für deinen kleinen Transformer. **Lernrate** bedeutet, grob gesprochen, wie gross die Änderungen ausfallen. AdamW passt Änderungen anhand vergangener Gradienten an; *weight decay* zieht bestimmte Gewichte zusätzlich etwas in Richtung null. Zunächst darfst du eine einfache gemeinsame Einstellung verwenden. Später kannst du begründet Parametergruppen bilden, etwa für Normierungsgewichte und Bias-Werte. [AdamW-Dokumentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).

**Gradient Clipping** begrenzt hier die gemeinsame Gradientennorm auf höchstens 1. Es ersetzt keine korrekte Lernrate und repariert keine falschen Daten. Wenn du später float16 mit GradScaler verwendest, müssen die Gradienten vor dem Clipping zuerst entskaliert werden.

### 7C Ein sehr kleines Beispiel absichtlich auswendig lernen

Bevor du längere Läufe startest, wiederholst du einige Hundert Updates auf **demselben kleinen Batch**. Deaktiviere für diesen Test Dropout und gegebenenfalls Weight Decay. Du willst wissen, ob Daten, Modell, Loss und Optimierer grundsätzlich zusammenarbeiten.

Für den ersten Bigram-Test eignet sich das eindeutige `abc`-Muster. Für den Transformer kannst du einen kleinen festen Batch aus einem kurzen Text verwenden. Achte darauf, dass nicht identische sichtbare Kontexte unterschiedliche Ziele verlangen. Sonst ist ein Loss nahe null grundsätzlich nicht erreichbar.

Notiere den Loss am Anfang und in festen Abständen. Er sollte deutlich sinken; ein exakter Zielwert ist nicht für jedes Beispiel sinnvoll. Wenn er sich überhaupt nicht bewegt, prüfe vor einem grösseren Modell:

- Enthält der Optimierer die Modellparameter?
- Sind die Gradienten nach `backward()` vorhanden und endlich?
- Ändert sich mindestens ein Parameter nach `optimizer.step()`?
- Sind `x` und `y` richtig verschoben und nicht leer?
- Läuft die Trainingsrechnung ohne `torch.no_grad()`?
- Ist die Lernrate versehentlich null oder extrem klein?

Absichtliches **Overfitting** bedeutet hier Auswendiglernen als Funktionstest. Es beweist noch keine Fähigkeit für unbekannte Texte. Nach diesem Test initialisierst du für deine eigentlichen Vergleichsläufe ein frisches Modell.

### 7D Training und Validierung auseinanderhalten

Im normalen Training ziehst du Batches nur aus den Trainingsdaten. In festen Abständen prüfst du eine feste Auswahl von Validierungsfenstern oder den ganzen überschaubaren Validierungsteil.

Für die Prüfung brauchst du beides:

```python
# Ausschnitt: Die Validierungs-Batches und Loss-Berechnung ergänzt du selbst.
model.eval()
with torch.no_grad():
    # Vorhersagen und Loss auf Validierungsdaten berechnen.
    pass
model.train()
```

`eval()` ändert beispielsweise Dropout-Verhalten. `no_grad()` schaltet die Gradientenverfolgung ab. Diese beiden Aufgaben sind verschieden. Für den mittleren Validierungs-Loss addierst du die Loss-Summen und teilst durch die Anzahl bewerteter Ziel-Tokens. Ein ungewichteter Mittelwert aus ungleich grossen Batches wäre ungenau.

Protokolliere Schrittzahl, verarbeitete Tokens, Trainings-Loss, Validierungs-Loss und Lernrate. Für spätere Vergleiche miss auch den Trainings-Loss auf einer festen Auswahl im `eval()`-Modus; der laufende Trainings-Loss mit Dropout ist sonst nicht direkt gleichartig.

| Beobachtung | Mögliche Bedeutung und Prüfung |
|---|---|
| Beide Loss-Werte sinken | Das Modell lernt auch Muster, die auf die zurückgehaltenen Daten passen. Trotzdem Datenüberschneidungen prüfen. |
| Training sinkt, Validierung steigt | Mögliches Overfitting. Früheren guten Stand vergleichen, Datenmenge und Modellgrösse prüfen. |
| Beide bleiben fast gleich | Lernrate, Zielverschiebung, Parameterupdates und Datensatz prüfen. |
| Plötzlich `NaN` oder `inf` | Letzten endlichen Schritt finden; Eingaben, Masken, Gradienten und Lernrate prüfen. Zunächst float32 verwenden. |

Validierungsdaten beeinflussen durch deine Modellentscheidungen indirekt das Projekt. Nutze einen späteren separaten Testteil deshalb erst für eine abschliessende Bewertung.

### 7E Erst nach einem funktionierenden Lauf erweitern

Ein **Lernratenplan** verändert die Lernrate während des Trainings. Eine spätere einfache Wahl: erst über einige Updates ansteigen lassen (*Warmup*), danach mit einer Cosinuskurve allmählich senken. Zum Beispiel könntest du 5 Prozent der geplanten Updates als Warmup und am Ende 10 Prozent der maximalen Lernrate erproben. Das sind Versuchsentscheidungen, keine feststehenden optimalen Werte. Prüfe den Plan als kleine Tabelle oder Kurve vor dem Lauf und speichere seinen Zustand mit.

**Gradient Accumulation** verarbeitet mehrere kleine Teilbatches vor einem Update. Bei `K` gleich grossen Teilbatches teilst du jeden mittleren Loss vor `backward()` durch `K`, setzt Gradienten nur am Anfang zurück und führst Clipping und Optimiererschritt erst nach allen Teilbatches aus. Die effektive Batchgrösse ist dann `B * K`. Mehr Kontextlänge erhältst du dadurch nicht.

**Mixed Precision** verwendet teilweise kleinere Zahlenformate. Erprobe später bfloat16 mit dem für dein Gerät unterstützten `torch.autocast`; halte den float32-Lauf als Vergleich. Ein halbierter Zahlenbedarf einzelner Tensoren bedeutet nicht, dass der ganze Trainingsspeicher halbiert wird. `torch.compile` ist ebenfalls eine spätere Optimierung: zuerst korrekte Ausgaben und Messwerte, dann Kompilierungsaufwand und mögliche Beschleunigung auf der konkreten Plattform prüfen.

**Weiter, wenn:** Ein eindeutiges Minibeispiel lernbar ist, die kausalen Tests bestehen und du getrennte Trainings- und Validierungswerte protokollieren kannst. Erst dann lohnt sich ein längerer Trainingslauf.

## Phase 8 Text erzeugen und Trainingsstände erhalten

**Ziel:** Dein Modell anwenden, kritisch beurteilen und nach einer Pause wiederverwenden. **Vorwissen:** Logits, Wahrscheinlichkeiten, Kontexte und gespeicherte Dateien. **Ablage:** `src/mintclm/inference/generate.py`, `evaluation/metrics.py`, `training/checkpoint.py`; Einstieg in `scripts/evaluate.py`; Tests in `tests/test_evaluation.py` und `tests/test_training.py`.

**Mindestziel:** Eine kurze Fortsetzung erzeugen, eine kleine Auswertung notieren und Gewichte in einem neuen Programmprozess wieder laden. **Später möglich:** Mehrere Sampling-Verfahren, automatische Vergleichsauswertungen und vollständige Trainingsfortsetzung.

### 8A Tokens nacheinander erzeugen

1. Einen nicht leeren Starttext mit genau dem zum Modell gehörenden Tokenizer codieren.
2. Die IDs als Tensor der Form `(1, Länge)` auf dasselbe Gerät wie das Modell bringen.
3. `model.eval()` verwenden und innerhalb von `torch.no_grad()` vorhersagen.
4. Höchstens die letzten `T` Tokens als Kontext verwenden.
5. Nur die **letzte Position** der Modellausgabe nehmen. Ihre Logits beschreiben das nächste Token.
6. Ein Token auswählen, anhängen und die Schritte wiederholen.
7. Nach einer festen Maximallänge oder beim eigenen End-Token stoppen.

Das Modell muss Text nicht auf einmal erzeugen. Du erweiterst die Eingabe nach jeder Vorhersage um genau ein Token. Bei deinen gelernten Positions-Embeddings werden die Positionen im jeweils verwendeten Kontextfenster von 0 an vergeben. Sehr lange oder übergreifende Zusammenhänge ausserhalb dieses Fensters stehen dem einfachen Modell nicht zur Verfügung.

**Greedy-Auswahl** nimmt immer den höchsten Logit. Sie ist einfach und kann schnell in Wiederholungen geraten. **Sampling** zieht ein Token zufällig gemäss den Wahrscheinlichkeiten. Bei Sampling beeinflusst eine **Temperatur** die Verteilung:

```python
# Ausschnitt: logits hat Form (B,T,V); temperature muss positiv sein.
temperature = 0.8
next_logits = logits[:, -1, :] / temperature
probs = torch.softmax(next_logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
```

Eine kleinere positive Temperatur betont die bevorzugten Tokens stärker; eine grössere macht die Auswahl gleichmässiger. Temperatur 0 behandelst du mit einem gesonderten Greedy-Zweig, statt durch null zu teilen. **Top-k** kann später vor Softmax alle bis auf die `k` bestbewerteten Tokens ausschliessen. Es muss mindestens ein Token übrig bleiben.

Für einen leeren Starttext brauchst du eine eigene, vorher trainierte Startkonvention, beispielsweise ein Begin-of-sequence-Token. Für den Einstieg gibst du einfach ein bekanntes Zeichen oder einen kurzen Startsatz ein.

Bei Byte- und Byte-BPE-Modellen können beliebig erzeugte Tokenfolgen ungültige UTF-8-Bytefolgen bilden. Für die Anzeige darfst du eine gesonderte Decodierung mit `errors="replace"` verwenden; Ersatzzeichen wie `�` zeigen dann ein Ergebnisproblem an. Die strenge Wiederherstellungsprüfung des Tokenizers bleibt unverändert. Decodiere vollständige Bytefolgen, nicht jeden Token einzeln.

### 8B Ergebnisse nachvollziehbar bewerten

Lege zum Beispiel fünf feste Starttexte fest und speichere jeweils eine kurze Fortsetzung. Notiere Modellstand, Temperatur, Sampling-Seed und maximale Länge. Bewahre auch schwache Beispiele auf. Nur die schönste Ausgabe auszuwählen würde den tatsächlichen Zustand verzerren.

Beurteile getrennt:

- **Messbare Vorhersagequalität:** Validierungs-Loss auf einem festgelegten Datenteil.
- **Lesbare Muster:** Entstehen erkennbare Wörter, Satzzeichen oder ein passender Stil?
- **Grenzen:** Wiederholungen, abgebrochene Wörter, widersprüchliche Aussagen, Ersatzzeichen oder offensichtliche Kopien.

Ein niedrigerer Loss garantiert nicht, dass jede einzelne Fortsetzung besser klingt. **Perplexity** ist `exp(mittlerer Cross-Entropy-Loss)`, sofern du den natürlichen Logarithmus und eine konsistente Bewertung verwendest. Sie ist eine weitere Darstellung derselben Vorhersagequalität; kleiner ist bei gleicher Aufgabe besser.

Vergleiche Token-Loss und Perplexity nur bei gleichem Tokenizer, gleichem Bewertungsinhalt und gleicher Zählweise direkt. Ein Byte- und ein BPE-Token decken unterschiedlich viel Text ab. Bei einem späteren Vergleich verschiedener Tokenizer kannst du die aufsummierten negativen Logwahrscheinlichkeiten zusätzlich auf die ursprüngliche UTF-8-Bytemenge beziehen: `bits_per_byte = nll_sum / (log(2) * anzahl_bytes)`. Dafür müssen Texte und Konventionen für Start, Ende und Kontext vergleichbar sein; einfach mittlere Token-Loss-Werte nebeneinanderzustellen reicht nicht.

**Prüfung:** Starte mit zehn neuen Tokens. Kontrolliere, dass die Eingabe je Schleifendurchlauf um eins wächst und nie mehr als `T` Tokens an das Modell gehen. Ein trainiertes `abc`-Bigram-Modell sollte das bekannte Muster oft fortsetzen.

**Typische Fehler:** Die erste statt der letzten Ausgabeposition verwenden; das neue Token nicht anhängen; Tokenizer verwechseln; `model.train()` bei Generierung mit Dropout aktiv lassen; ein Startzeichen verwenden, das dein Zeichen-Tokenizer nicht kennt. Prüfe zuerst die IDs und die Eingabelänge.

### 8C Gewichte und Trainingsstand unterscheiden

Zum reinen Erzeugen brauchst du mindestens Modellgewichte, Architekturkonfiguration und genau den passenden Tokenizer. Zur sinnvollen Trainingsfortsetzung brauchst du zusätzlich den Zustand des Optimierers sowie Schrittzähler und gegebenenfalls Lernratenplan und GradScaler. Das erneute Laden von Gewichten allein stellt AdamWs Verlauf nicht wieder her. [PyTorch zum Speichern und Laden](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Dieses **Muster** setzt vorhandene Objekte und eine Konfiguration als Dictionary aus einfachen Zahlen, Zeichenketten und Listen voraus:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "config": config_dict,
    "tokenizer_id": tokenizer_hash,
}
torch.save(checkpoint, checkpoint_path)
```

`state_dict()` ist ein Dictionary mit den gespeicherten Zuständen. Ein Hash der Tokenizer-Datei hilft dir zu erkennen, ob genau dieselben Regeln verwendet werden. Speichere zusätzlich die Tokenizer-Datei selbst, die Datenaufteilung beziehungsweise deren Kennung und die verwendeten Softwareversionen. Ein Hash ersetzt diese Inhalte nicht.

Zum Laden baust du deine Modellklasse mit der gespeicherten Konfiguration erneut auf und übernimmst die Gewichte:

```python
# Muster: MintCLM und der gerätespezifische Aufbau werden von dir implementiert.
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model = MintCLM(**checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model.eval()
```

Der Klassenname ist hier ein Vorschlag; dieses Beispiel läuft erst, wenn deine Klasse diese Konfigurationsfelder akzeptiert. Speichere einfache Dictionaries und Tensoren statt das gesamte Python-Modellobjekt. Das macht Änderungen und Gerätewechsel leichter nachvollziehbar. Lade für diese Übungen deine eigenen Checkpoints; die Herkunft fremder Dateien muss vor dem Laden geklärt sein.

Für Fortsetzung setzt du das Modell auf das Zielgerät, erzeugst den passenden Optimierer und lädst seinen Zustand ebenfalls. Prüfe die Geräte seiner Zustands-Tensoren und stelle `model.train()` ein. Lade gegebenenfalls den Lernratenplan und verwende als nächsten Schritt den korrekt fortgesetzten Zähler.

**Für möglichst genaue Fortsetzung** kommen Zufallszustände von Python, NumPy und PyTorch inklusive CUDA sowie die Position oder der Zustand der Batch-Auswahl hinzu. Speichere anfangs nur direkt nach einem abgeschlossenen Optimiererschritt. Dann musst du keine halb angesammelten Gradienten erhalten. Ein Seed allein beschreibt nicht, wie weit der Zufallszahlengenerator bereits gelaufen ist. Selbst mit diesen Zuständen sind identische Ergebnisse über verschiedene Geräte und Softwareversionen nicht allgemein garantiert. [PyTorch zur Reproduzierbarkeit](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

Schreibe einen Checkpoint zunächst vollständig in eine temporäre Datei im selben lokalen Ordner und ersetze danach den endgültigen Namen. Das verringert das Risiko, einen bestehenden guten Stand durch einen unterbrochenen Schreibvorgang zu verlieren. Bewahre für längere Läufe mindestens den neuesten und den besten Validierungsstand getrennt auf.

**Prüfung:** Vergleiche vor und nach dem Laden auf derselben CPU und im `eval()`-Modus die Logits derselben Eingabe mit `torch.testing.assert_close`. Führe ausserdem nach dem Laden einen weiteren Trainingsschritt aus. Der ausführlichere Fortsetzungstest steht im nächsten Testkapitel.

**Weiter, wenn:** Du kannst in einem neuen Prozess mit geladenen Gewichten dieselben Vorhersagen berechnen und die verwendete Modell- und Tokenizerkonfiguration eindeutig zuordnen.

## Phase 9 BPE Schritt für Schritt selbst implementieren

**Ziel:** Häufige Bytefolgen bekommen gemeinsame Tokens. **Vorwissen:** Byte-Tokenisierung, Dictionaries, Zählen, Slicing und Dateispeicherung. **Ablage:** `src/mintclm/tokenizer/bpe.py`, `scripts/train_tokenizer.py`, `tests/test_tokenizer.py`; Tokenizer-Dateien im lokalen `tokenizers/`-Ordner. Kleine Versuchsergebnisse können ins Repository.

**Mindestziel:** Eine einfache, deterministische Byte-BPE-Implementierung auf wenigen kurzen Dokumenten. **Später möglich:** Schnellere Paarzählung, Verarbeitung grösserer Korpora und ein kontrollierter Vergleich mit dem Byte-Modell.

### 9A Die Grundidee von Hand nachvollziehen

**Byte Pair Encoding** beginnt hier mit 256 Tokens: eines pro Byte. Danach fasst du häufige benachbarte Token-Paare zu neuen Tokens zusammen. So kann ein oft vorkommendes Wortstück in einem Schritt dargestellt werden.

Für das ASCII-Wort `banana` kannst du die Bytes zunächst wie Zeichen lesen:

```text
Start:             b | a | n | a | n | a
Häufiges Paar:          a n       a n
Nach einem Merge:  b | an | an | a
```

`an` und `na` kommen je zweimal vor. Du brauchst also eine **feste Gleichstandsregel**. Eine mögliche Wahl ist das lexikografisch kleinste Paar von IDs: Zuerst wird die erste Zahl verglichen, bei Gleichstand die zweite. Bei den ASCII-IDs gewinnt dann `(a,n)`. Ein *Merge* ist eine Zusammenführungsregel, beispielsweise `(97,110) → 256`.

Das neue Token 256 bedeutet die Bytefolge `[97,110]`. Bei späteren Merges können auch schon zusammengeführte Tokens beteiligt sein. Die Bedeutung eines neuen Tokens ist immer die aneinandergehängte Bytefolge seiner beiden Vorgänger.

Guter nächster Lerntext: der [BPE-Kurs von Hugging Face](https://huggingface.co/learn/llm-course/chapter6/5). Er erläutert den Algorithmus; du verwendest ihn als Erklärung und schreibst deine eigene Umsetzung. Die ursprüngliche Anwendung auf Wortteile findest du später in [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/).

### 9B Kleine Funktionen statt eines grossen Programms schreiben

Baue und prüfe nacheinander:

| Funktion oder Methode | Aufgabe | Einfacher Kontrollfall |
|---|---|---|
| `count_pairs(sequences)` | Benachbarte ID-Paare je Dokument zählen | `[1,2,1,2]` enthält `(1,2)` zweimal und `(2,1)` einmal. |
| `choose_pair(counts)` | Häufigstes Paar mit fester Gleichstandsregel wählen | Zwei gleich häufige Paare liefern immer dieselbe Wahl. |
| `merge_pair(ids, pair, new_id)` | Ein Paar links nach rechts ohne Überlappung ersetzen | `[1,1,1]` wird bei `(1,1)→256` zu `[256,1]`. |
| `train(documents, num_merges)` | Wiederholt zählen, wählen, ersetzen und Regel speichern | Gleicher Korpus und gleiche Einstellungen liefern gleiche Regeln. |
| `encode(text)` | Neue Texte mit den bereits gelernten Regeln codieren | Es werden keine neuen Regeln gelernt. |
| `decode(ids)` | Bytefolgen verbinden und einmal als UTF-8 decodieren | Text aus `encode` wird exakt wiederhergestellt. |

Wähle anfangs `num_merges=10` oder 20 und drucke jede Regel aus. So kannst du nachvollziehen, was dein Code tut.

### 9C Den Trainingsalgorithmus festlegen

1. Wandle jedes **Trainingsdokument** separat mit `text.encode("utf-8")` in Byte-IDs um.
2. Lege für IDs 0 bis 255 die Bedeutung als jeweiliges einzelnes Byte fest.
3. Zähle benachbarte Paare innerhalb der Dokumente. Zähle kein Paar über eine Dokumentgrenze hinweg.
4. Wähle das häufigste Paar anhand deiner festen Gleichstandsregel.
5. Vergib eine neue ID und speichere sowohl die Regel als auch deren Position in der Reihenfolge, den **Merge-Rang**.
6. Ersetze dieses Paar in allen Trainingsdokumenten links nach rechts ohne überlappende Ersetzungen.
7. Wiederhole die Schritte bis zur gewählten Zahl von Merges oder bis kein Paar mehr vorhanden ist. Du kannst auch ausdrücklich festlegen, bei Häufigkeit kleiner als 2 zu stoppen.

Bei `[a,a,a,a]` wird das Paar `(a,a)` beim Zählen an drei benachbarten Positionen gefunden. Eine nicht überlappende Ersetzung ergibt trotzdem nur `[aa,aa]`. Zählen und Ersetzen haben hier unterschiedliche Aufgaben; dafür brauchst du einen eigenen Test.

Diese einfachste Variante darf innerhalb eines Dokuments auch über Leerzeichen hinweg zusammenführen. Das ist eine bewusste Designentscheidung. Eine spätere Vorzerlegung in Wörter oder Textabschnitte verändert das Verfahren und muss bei Training und `encode` gleich sein.

### 9D Neue Texte richtig codieren

Beginne wieder mit den UTF-8-Bytes des neuen Textes. Wende die **gespeicherten** Regeln in ihrer gelernten Reihenfolge an, jeweils mit derselben nicht überlappenden Ersetzung. Das ist eine leicht nachvollziehbare erste Encoder-Implementierung. Eine spätere effizientere Variante sucht jeweils das vorhandene Paar mit dem kleinsten Merge-Rang.

Du zählst auf dem neuen Text nicht nochmals frei die häufigsten Paare und lernst den Tokenizer nicht bei jedem Aufruf neu. Sonst würden dieselben IDs je nach Text etwas anderes bedeuten. Trainiere die Regeln auch nicht mit Validierungstexten.

Weil alle 256 Bytes im Anfangsvokabular enthalten bleiben, kann dein Encoder auch unbekannte Wörter und gültige UTF-8-Zeichen darstellen. Häufige Folgen sind kürzer codiert, andere bleiben mehrere einzelne Bytes. Zur Decodierung verbindest du zuerst alle Token-Bytefolgen und decodierst dann die gesamte Bytefolge. Ein BPE-Token muss allein kein vollständiges Unicode-Zeichen ergeben.

### 9E Regeln dauerhaft speichern

Speichere in einem klaren Format, beispielsweise JSON:

- Formatversion und die Festlegung «UTF-8, Basis-IDs 0 bis 255».
- Geordnete Liste der Merges mit linker ID, rechter ID und neuer ID.
- Token-Bedeutungen als Listen von Bytewerten oder eine eindeutig rekonstruierbare Merge-Tabelle.
- Gewählte Anzahl Merges, tatsächlich erzeugte Anzahl und Gleichstandsregel.
- Sondertoken-IDs und deren gesonderte Behandlung, falls du welche einführst.
- Kennung der Trainingsdaten und der Bereinigung.

Für die erste Version brauchst du keine Sondertokens. Wenn du nach `M` tatsächlich erzeugten Merges ein End-Token hinzufügst, kannst du ihm ID `256 + M` geben. Die Gesamtgrösse ist dann `257 + M`. Dieses End-Token hat keine normale Bytebedeutung und wird bei der Textdecodierung nach deiner ausdrücklich gewählten Regel behandelt. Seine ID darf mit keinem Merge kollidieren.

### 9F Prüfen und mit dem Modell verbinden

Prüfe zuerst Wiederherstellung, deterministische Regeln, überlappende Paare, leeren Text, neue Wörter, Umlaute und Emoji. Nach Speichern und Laden müssen sowohl die IDs als auch die decodierten Texte übereinstimmen. Das gesamte Vokabular muss bekannt sein, einschliesslich reservierter Sondertokens.

Miss dann auf Training **und** Validierung, wie viele Tokens derselbe Text benötigt. Ein einfaches Mass ist `Anzahl UTF-8-Bytes / Anzahl BPE-Tokens`: Ein grösserer Wert bedeutet mehr Bytes je Token. Behandle leere Texte separat, damit du nicht durch null teilst. Eine bessere Kompression beweist noch keine bessere Sprachmodellqualität; ein grösseres Vokabular vergrössert auch Embedding und Ausgabeschicht.

Wenn du den Tokenizer wechselst, tokenisierst du die Daten neu und passt die Modell-Vokabulargrösse an. Trainiere für den ersten Vergleich ein frisches Modell. Alte Gewichte lassen sich bei veränderter Tokenbedeutung nicht einfach passend umdeuten.

**Typische Fehler:** Paarzahlen nach einem Merge nicht neu berechnen; überlappende Ersetzungen; Regeln in einer ungeordneten Datenstruktur verlieren; Wörter beim Decodieren mit zusätzlichen Leerzeichen verbinden; Bytefolgen zu früh decodieren. Prüfe dafür kurze Listen, die du von Hand berechnen kannst.

**Weiter, wenn:** Du kannst die ersten Merges erklären, beliebige gültige Testtexte verlustfrei codieren und mit einem erneut geladenen Tokenizer dieselben IDs erzeugen.

## Phase 10 Speicher und Architektur gezielt erweitern

**Ziel:** Eine konkrete Grenze oder Forschungsfrage untersuchen. **Vorwissen:** Ein getesteter kleiner Transformer, eine dokumentierte Datenaufteilung und ein funktionierender Ausgangslauf. **Ablage:** Je nach Änderung die vorhandenen Modelldateien, `data/dataset.py`, `training/sft.py`, `scripts/finetune.py` oder `app/app.py`; Vergleiche in `experiments/ablations/` und `experiments/results/`.

**Mindestziel:** Du darfst nach dem kleinen Modell zunächst aufhören und dein Ergebnis auswerten. Eine Erweiterung ist für diese Etappe optional. **Später möglich:** Genau eine Änderung auswählen, begründen und gegen den Ausgangszustand prüfen.

### Wann numpy.memmap nützlich wird

Der Arbeitsspeicher des Computers, **RAM**, hält deine Python-Daten. Der Grafikspeicher, **VRAM**, hält auf einer gewöhnlichen separaten GPU Modell und gerade benötigte Rechendaten. Ein grosser Gesamtdatensatz muss nicht vollständig in den VRAM passen; normalerweise überträgst du nur den nächsten Batch.

Für einige Tausend oder Millionen Tokens ist ein gewöhnliches Array oft einfach genug. `numpy.memmap` wird interessant, wenn das vollständige Token-Array nicht mehr bequem in den RAM passt oder wenn viele Starts und mehrere Prozesse eine grosse Datei gezielt lesen sollen. Es erlaubt den Zugriff auf Ausschnitte einer Datei, ohne zunächst alles einzulesen. Der tatsächlich genutzte Speicher hängt auch vom Betriebssystem-Cache und Zugriffsmuster ab. [NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

Die unkomplizierte Zwischenstufe ist eine `.npy`-Datei:

```python
# Ausschnitt: ids_array ist dein bereits erzeugtes NumPy-Array.
import numpy as np

np.save(token_file, ids_array)
loaded = np.load(token_file, mmap_mode="r")
small_window = loaded[100:133]
```

`.npy` speichert Datentyp und Form mit. Ein rohes `.bin` für `np.memmap` braucht diese Angaben zusätzlich, inklusive Byte-Reihenfolge und Dateiformat. Falsche Metadaten können scheinbar lesbare, aber falsche IDs ergeben.

Wähle den Datentyp nach der **höchsten tatsächlich möglichen ID**: `uint16` reicht für IDs 0 bis 65535, einschliesslich aller Sondertokens. Sonst brauchst du beispielsweise `uint32`. Prüfe den Wertebereich **vor** einer Umwandlung; ein zu kleiner Datentyp kann Werte verändern. Für Embedding und Cross-Entropy wandelst du den jeweiligen Batch in `torch.long` um. Wenn du eine schreibgeschützte Memmap verwendest, erstelle für den kleinen Batch eine eigene Kopie, bevor du ihn als veränderbaren PyTorch-Tensor nutzt.

Zur Grössenordnung: 100 Millionen IDs belegen als `uint16` ungefähr 200 Millionen Bytes, als `uint32` ungefähr 400 Millionen Bytes, jeweils ohne weitere Dateistrukturen. Das ist eine Speicherrechnung, keine Empfehlung für deinen ersten Datensatz.

**Prüfung:** Vergleiche Anfang, Ende und zufällige Fenster mit dem ursprünglichen Array, prüfe Tokenanzahl, Dateiformat und maximale ID. Miss danach, ob das Laden wirklich zum Engpass wird. **Typische Fehler:** Rohdatei mit falschem Datentyp lesen, gleichzeitig dieselbe Datei schreiben und trainieren, den gesamten Memmap-Inhalt doch wieder kopieren. **Weiter, wenn:** Deine Batches unverändert sind und der neue Ablauf den tatsächlichen Speicher- oder Ladebedarf verbessert. Memmap beschleunigt nicht automatisch die Modellrechnung und vergrössert nicht den VRAM.

### Mögliche Änderungen mit einer klaren Prüffrage

| Erweiterung | Was sich verändert | Was du vorher lernen solltest | Klein anfangen und prüfen |
|---|---|---|---|
| RMSNorm | Normiert anhand der mittleren quadrierten Grössen ohne Mittelwert-Abzug | Mittelwert, Quadratwurzel und letzte Tensorachse | Eigene Funktion an einem kleinen Vektor durchrechnen und bei `eps > 0` auf endliche Ausgaben prüfen. Dann gleiche Trainingsaufgabe vergleichen. |
| RoPE | Positionsinformation wird durch Drehungen von Q und K eingebracht | Paarweise Koordinaten, Sinus und Cosinus | Zwei Zahlenpaare bei Position 0 und 1 prüfen; gerade Kopfbreite und konsistente Positionen sicherstellen. Danach die bisherigen Positions-Embeddings im geplanten Vergleich ersetzen. |
| GQA | Mehrere Query-Köpfe teilen sich Key- und Value-Köpfe | Multi-Head-Achsen und deren Zuordnung | Zuerst gleich viele Q- und KV-Köpfe wie bisher; danach beispielsweise 4 Q- und 2 KV-Köpfe. Prüfe Teilbarkeit und Gleichheit der Ausgangsvariante. |
| SwiGLU | Ein Zweig steuert einen anderen im Feedforward-Netz durch elementweises Multiplizieren | Lineare Schichten und SiLU-Aktivierung | Form jeder Projektion prüfen; Parameterzahl gegenüber dem alten Feedforward-Netz mit angeben. |
| Grösseres Modell | Mehr lernbare Kapazität und höherer Speicherbedarf | Laufzeit- und Speichermessung | Nur eine Grössenachse erhöhen und einen kurzen vollständigen Trainingsversuch messen. |
| SFT | Das Modell lernt aus Beispielen gewünschter Eingaben und Antworten | Datenformat, Spezialtokens, maskierter Loss und getrennte Auswertung | Einige selbst geschriebene Frage-Antwort-Paare prüfen; nur die vorgesehenen Antwort-Zielpositionen bewerten. |
| Lokale Oberfläche | Du kannst das trainierte Modell bequem ausprobieren | Laden, Generierung und Abbruchbedingungen | Erst eine lokale Eingabe mit Maximalzahl neuer Tokens; prüfen, dass wirklich dein gespeichertes Modell und dein Tokenizer geladen werden. |

Die Details dieser Architekturideen findest du in den jeweiligen Originalarbeiten: [RMSNorm](https://arxiv.org/abs/1910.07467), [RoPE](https://arxiv.org/abs/2104.09864), [GQA](https://arxiv.org/abs/2305.13245) und [GLU-Varianten mit SwiGLU](https://arxiv.org/abs/2002.05202). Du musst diese Arbeiten heute noch nicht vollständig lesen.

GQA spart insbesondere beim späteren **KV-Cache**: Dieser speichert bereits berechnete Keys und Values während der schrittweisen Generierung. Eine tatsächliche Beschleunigung hängt auch von der Umsetzung und Hardware ab. Einen solchen Cache brauchst du für die erste Generierung noch nicht.

**Supervised Fine-Tuning** ist weiterhin ein Nächstes-Token-Training, nun an vorgegebenen Beispielen gewünschten Verhaltens. Wenn du nur Antwortteile bewerten möchtest, setzt du die übrigen Zielpositionen auf den von deiner Loss-Funktion ignorierten Wert, beispielsweise `-100`. Prompt-Tokens bleiben als Eingabe sichtbar. Prüfe die Zuordnung nach der Zielverschiebung, besonders den ersten Antworttoken und das Antwortende. Padding muss ebenfalls korrekt aus dem Loss entfernt werden; je nach Packstrategie brauchst du zusätzlich eine passende Attention-Maske. Ein Batch darf nicht ausschliesslich ignorierte Ziele enthalten.

Ein wenig SFT kann ein sehr schwaches Grundmodell nicht in einen zuverlässigen Assistenten verwandeln. Eine Oberfläche macht seine vorhandenen Fähigkeiten zugänglich; sie ersetzt keine Auswertung.

Für jeden Vergleich notierst du: **Fragestellung, genau eine geplante Änderung, Daten und Tokenizer, Parameterzahl, Trainingsbudget, Ergebnis und Unsicherheit.** Gleiche Trainingsschritte allein bedeuten bei unterschiedlichen Batchgrössen nicht gleich viele gesehene Tokens. Bei knappen Ressourcen reicht ein kleiner sauber beschriebener Vergleich; bei schwankenden Ergebnissen sind später mehrere Seeds hilfreich.

**Weiter, wenn:** Du kannst sagen, welche konkrete Beobachtung für oder gegen die Änderung spricht. Eine bessere Zahl in einem einzelnen Lauf ist noch kein allgemeiner Beweis, dass ein Baustein immer überlegen ist.

## Tests und Fehlersuche als Teil deiner Arbeit

Ein Test ist eine kleine Behauptung über das erwartete Verhalten. Mit `assert` hältst du diese Behauptung fest. Mit `pytest` kannst du viele solche Prüfungen automatisch ausführen. Beginne mit wenigen besonders aussagekräftigen Tests und ergänze neue, wenn du einen Baustein hinzufügst oder einen Fehler gefunden hast.

### Die wichtigsten Prüfungen

| Prüfung | Erwartung | Was ein Fehlschlag bedeuten könnte |
|---|---|---|
| Tokenizer-Rundreise | `decode(encode(text)) == text` | Leerzeichen verloren, falsche Merge-Reihenfolge, defekte Bytebehandlung oder veränderte Normalisierung. |
| Tokenizer speichern und laden | Genau gleiche IDs für denselben Text | Tabelle oder Regelreihenfolge wurde nicht vollständig gespeichert. |
| Zielverschiebung | Jedes `y[t]` ist das nach `x[t]` folgende Token | Um eins verrutschter Ausschnitt, doppeltes Verschieben oder Grenzfehler. |
| Tensorformen | Eingabe `(B,T)`, Ausgabe `(B,T,V)` | Achsen vertauscht, Köpfe falsch zusammengefügt oder Vokabulargrösse verwechselt. |
| Attention-Gewichte | Summe je Zeile 1 und rechts der Diagonale 0 | Falsche Softmax-Achse oder falsche Maske. |
| Kausalität des ganzen Modells | Zukünftige Eingabe verändert frühere Logits nicht | Attention-Leck oder ein anderer Baustein vermischt Zeitpositionen unerlaubt. |
| Lernfähigkeit | Loss auf einem eindeutigen Minibeispiel sinkt deutlich | Parameterupdates, Gradienten, Daten oder Lernrate stimmen nicht. |
| Checkpoint | Logits nach dem Laden stimmen überein | Falsche Architektur, Gewichte, Tokenizer oder unterschiedlicher Modus. |
| Fortsetzung | Ein weiterer Schritt ist möglich; unter kontrollierten Bedingungen entspricht er dem ununterbrochenen Schritt | Optimierer, Zufallszustände, Scheduler oder nächste Datenposition fehlen. |

### Einen aussagekräftigen Kausalitätstest schreiben

Der Test verwendet dein fertiges Modell auf der CPU, eine Kontextlänge von mindestens 8 und ein Vokabular mit mindestens zwei Tokens. Er ist ein Muster für `tests/test_model.py`; die Erzeugung des passenden `model` ergänzt du selbst.

```python
import torch

model.eval()
torch.manual_seed(12)
vocab_size = 256
x = torch.randint(0, vocab_size, (2, 8))
changed = x.clone()
changed[:, 4:] = (changed[:, 4:] + 1) % vocab_size

with torch.no_grad():
    original_logits = model(x)
    changed_logits = model(changed)

torch.testing.assert_close(
    original_logits[:, :4],
    changed_logits[:, :4],
    rtol=1e-5,
    atol=1e-6,
)
```

Passe `vocab_size` an deine Konfiguration an. Die ersten vier Eingabe-Tokens bleiben gleich, alle späteren werden verändert. Die Logits an Position 0 bis 3 müssen trotzdem gleich bleiben. Auch Position 3, die Token 4 **vorhersagt**, darf das tatsächliche Eingabetoken an Position 4 nicht sehen.

Verwende `eval()`, damit Dropout den Vergleich nicht zufällig verändert. Prüfe mehrere Eingaben und Grenzpositionen. Ergänze einen Vergleich zwischen einem kurzen Präfix allein und demselben Präfix am Anfang einer längeren Eingabe. Ein Modell ohne Attention könnte den Kausalitätstest ebenfalls bestehen; zusammen mit dem direkten Maskentest und dem Lernversuch wird die Prüfung aussagekräftiger.

### Einen Checkpoint wirklich testen

1. Trainiere ein paar kleine Schritte und wähle eine feste Eingabe.
2. Berechne deren Logits in `eval()` und speichere den Trainingsstand.
3. Erzeuge ein neues Modell und einen neuen Optimierer; lade den Stand.
4. Vergleiche die Logits auf demselben Gerät mit `assert_close`.
5. Führe einen weiteren Schritt aus und prüfe endlichen Loss und veränderte Gewichte.
6. Als spätere strengere Prüfung: Vergleiche «speichern, laden, nächster fester Batch» mit «ohne Unterbruch derselbe nächste Batch». Verwende zunächst CPU, Dropout 0 und identische Einstellungen. Für stochastische Abläufe musst du zusätzlich Zufallszustände wiederherstellen.

### Tests praktisch starten

Eine Testdatei enthält beispielsweise Funktionen mit Namen wie `test_byte_roundtrip`. Beim ersten Aufbau kannst du viele Prüfungen noch direkt im Lernskript mit `assert` durchführen.

```powershell
Set-Location -LiteralPath 'C:\Users\wunde\MintCLM\MintCLM 1.0'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest tests/test_tokenizer.py
```

Die `PYTHONPATH`-Zeile macht dein `src`-Paket für diese Konsole auffindbar. Sie ist ein einfacher Übergang für die vorhandene Struktur. Für `pip install -e .` braucht das Projekt später erst eine passende Paketkonfiguration wie `pyproject.toml`. Die bislang leeren Testdateien enthalten noch keine Prüfungen; «keine Tests gefunden» ist deshalb kein Nachweis korrekter Funktionen. Führe jeweils die Tests aus, die du bereits umgesetzt hast.

Wenn etwas fehlschlägt, verkleinere auf **ein Beispiel, einen Block und die CPU**. Drucke Formen, Datentypen, Geräte, minimale und maximale IDs sowie den ersten fehlerhaften Zwischenschritt. Lies die eigentliche Fehlermeldung am Ende der Ausgabe und suche dann die erste Stelle in deinem eigenen Code. Ändere jeweils nur eine Vermutung und notiere, was sich dadurch verändert hat.

Nach einem GPU-Fehler kann der laufende Prozess in einem unbrauchbaren Zustand sein. Starte ihn für einen sauberen erneuten Versuch neu. Hebe eine kleine Eingabe auf, die den Fehler zuverlässig reproduziert; das ist oft hilfreicher als eine grosse Menge zusätzlicher Protokolle.

## Zeit und Hardware realistisch planen

### Ein flexibler Plan für drei bis sechs Monate

**Planungsannahme:** Im Durchschnitt stehen dir etwa drei bis fünf aktive Projektstunden pro Woche zur Verfügung. Du kannst sie beispielsweise auf eine längere und zwei kurze Sitzungen verteilen. Das ist eine Annahme zum Planen, keine Erwartung an dich. Installation, Fehlersuche und Schulwochen können unterschiedlich viel Zeit beanspruchen.

Ein möglicher roter Faden über ungefähr 20 Wochen:

| Zeitraum | Schwerpunkt | Sichtbares Ergebnis |
|---|---|---|
| Woche 1 | Einrichtung und erster Textversuch | Ein Skript läuft; Text wird in IDs und zurück verwandelt. |
| Wochen 2 bis 4 | NumPy, Tensoren und Gradienten | Kleine selbst erklärte Übungen und ein lernender Zahlenparameter. |
| Wochen 5 bis 6 | Kleine Daten, Aufteilung, Tokenizer und Fenster | Ein geprüfter Datensatz mit getrennten Teilen. |
| Woche 7 | Bigram-Modell | Ein einfaches Muster wird gelernt und fortgesetzt. |
| Wochen 8 bis 11 | Eigene Attention und Transformer | Formen und Kausalität sind geprüft. |
| Wochen 12 bis 14 | Kurze Trainingsläufe und Validierung | Lernkurven oder Tabellen, feste Textproben und Checkpoints. |
| Wochen 15 bis 16 | Laden, Auswertung und offene Fehler | Ein anderer Programmstart kann den Stand weiterverwenden. |
| Wochen 17 bis 18 | Optional BPE oder eine andere einzelne Frage | Ein eigener Tokenizer oder ein sauber begrenzter Vergleich. |
| Wochen 19 bis 20 | Ergebnisse erklären und dokumentieren | Nachvollziehbarer Projektbericht mit Grenzen und nächsten Ideen. |

Journal und kurze Zusammenfassungen schreibst du während der ganzen Zeit. Die letzten Wochen sind zum Ordnen und Erklären gedacht; die Dokumentation beginnt nicht erst dann.

Bei etwa zwei bis drei Stunden pro Woche kannst du die Lernphasen über sechs Monate verteilen und Erweiterungen weglassen. Bei mehr Zeit kannst du manche Phasen zusammenziehen und in ungefähr drei Monaten einen kleinen vollständigen Ablauf erreichen. Verdopple nicht automatisch Modell oder Daten, nur weil du schneller vorankommst.

Plane nach jeder zwei- bis dreiwöchigen Etappe eine Entscheidung: **Verstehe ich genug, macht mir die Arbeit noch Freude und was möchte ich als Nächstes untersuchen?** Wenn du langsamer vorankommst, verkleinere zuerst den Umfang. Ein gut erklärter kleiner Transformer ist ein klarer möglicher Abschluss. BPE, SFT und eine Oberfläche können getrennte spätere Etappen bleiben.

### Die Geräte nach dem konkreten Versuch auswählen

| Gerät | Sinnvoller möglicher Einsatz | Vorher klären |
|---|---|---|
| Laptop mit RTX 5060 | Entwicklung, Tests und erste kleine Trainingsläufe | Freier Grafikspeicher, funktionierender PyTorch-Build, Verhalten bei längerem Lauf. |
| Workstation mit RTX PRO 5000 | Längere oder grössere Versuche nach einem erfolgreichen Pilotlauf | Exakte GPU-Ausführung, Speicher, Betriebssystem, Zugang und verfügbare Zeit. |
| DGX Spark mit 128 GB gemeinsamem Speicher | Versuche, für die mehr gemeinsam nutzbarer Speicher interessant ist | ARM64-Umgebung, tatsächlich verfügbarer Speicher, Container und gemessene Geschwindigkeit. |
| Eventuelle H200 | Spätere Versuche bei tatsächlich verfügbarem Zugang | Konkrete Ausstattung, Laufzeitfenster, Umgebung und Datenübertragung. |

Es gibt hier bewusst keine behauptete Trainingsgeschwindigkeit für diese Geräte. Der Name einer GPU allein sagt nicht, wie schnell **dein** Programm läuft. Nutzungszeiten und Gerätekonfigurationen sind noch offen.

**Speicherkapazität** beantwortet: Passt der geplante Versuch hinein? **Geschwindigkeit** beantwortet: Wie lange dauert er? Dazu tragen unter anderem Rechenleistung, Speicherbandbreite, Datenladen, Zahlenformat und die verwendeten Rechenverfahren bei. Ein Gerät mit mehr Speicher ist deshalb nicht automatisch bei jedem kleinen Modell schneller.

Die 128 GB des DGX Spark werden zwischen CPU, GPU und weiteren Systemaufgaben geteilt. Es ist kein zusätzlich zum übrigen Arbeitsspeicher exklusiv verfügbarer VRAM-Pool. Ebenso sind Werbeangaben zur Ausführung grosser quantisierter Modelle keine Zusage, dass dieselbe Parameterzahl vollständig trainierbar ist. Vollständiges Training benötigt weitere Zustände und Zwischenergebnisse. [NVIDIA-Hardwareübersicht](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

### Speicherbedarf grob verstehen

Für einen einfachen AdamW-Lauf mit float32 kannst du als erste Rechnung ungefähr **16 Bytes pro Parameter** ansetzen: 4 für das Gewicht, 4 für den Gradienten und 8 für die beiden Adam-Zustände. Dazu kommen Aktivierungen, temporäre Rechenfelder, Daten und weitere Verwaltungsanteile. Diese Rechnung ist eine grobe Ausgangsbasis, kein vollständiges Speicherbudget.

Die **Aktivierungen** sind Zwischenergebnisse, die beim Rückwärtsrechnen gebraucht werden. Sie hängen stark von Batchgrösse und Kontextlänge ab. Deine einfache selbst implementierte Attention bildet ein Feld `(B,H,T,T)`; bei doppeltem `T` enthält dieses viermal so viele Einträge. Optimierte Attention-Verfahren können mit weniger gespeicherten Zwischenfeldern auskommen. Deshalb muss der Speicherbedarf mit der tatsächlichen Implementierung gemessen werden.

Bei einem Speicherfehler verkleinerst du zuerst Batchgrösse und gegebenenfalls Kontext. Gradient Accumulation kann später die gewünschte effektive Batchgrösse erhalten. Ein grösserer Datensatz auf der SSD allein verursacht noch keinen entsprechend grösseren VRAM-Bedarf, wenn nur einzelne Batches übertragen werden.

### Einen kleinen vollständigen Pilotlauf messen

Miss auf jedem Gerät dieselbe klar beschriebene Konfiguration. Ein Vorwärtslauf allein ist für eine Trainingsabschätzung zu wenig: Rückwärtslauf und Optimiererschritt gehören dazu. Verwende auch dasselbe Zahlenformat und dieselbe Art des Datenladens wie im geplanten Lauf.

1. Wähle Modell, `B`, `T`, Tokenizer, Optimierer und Zahlenformat.
2. Führe etwa 10 bis 20 vollständige Updates als Aufwärmphase aus. Adam-Zustände und gegebenenfalls Kompilierung werden dabei angelegt.
3. Miss anschliessend beispielsweise 50 bis 100 vollständige Updates. Falls die Messung extrem kurz ist, wähle mehr Updates.
4. Synchronisiere die GPU vor Start und Ende der Zeitmessung. GPU-Aufträge werden sonst häufig nur eingereiht, und die gemessene Zeit wäre irreführend. [PyTorch zu asynchroner CUDA-Ausführung](https://docs.pytorch.org/docs/stable/notes/cuda.html).
5. Notiere Zeit, Zahl der verarbeiteten Ziel-Tokens und maximalen Speicherbedarf. Prüfe zusätzlich die gesamte Gerätebelegung, etwa mit `nvidia-smi`.

Dieses **Messmuster** setzt voraus, dass `get_train_batch()` einen Batch liefert, `train_step(x, y)` genau ein vollständiges Update ausführt und `device`, `B` sowie `T` deine echten Einstellungen enthalten. Es verwendet feste Fenster ohne Padding und ohne Gradient Accumulation:

```python
import time
import torch

use_cuda = device.type == "cuda"
for _ in range(10):
    x, y = get_train_batch()
    train_step(x, y)

if use_cuda:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

steps = 50
start = time.perf_counter()
for _ in range(steps):
    x, y = get_train_batch()
    train_step(x, y)
if use_cuda:
    torch.cuda.synchronize(device)

seconds = time.perf_counter() - start
targets = steps * B * T
print("Sekunden pro Update:", seconds / steps)
print("Ziel-Tokens pro Sekunde:", targets / seconds)
if use_cuda:
    print("Peak belegte GiB:", torch.cuda.max_memory_allocated(device) / 2**30)
    print("Peak reservierte GiB:", torch.cuda.max_memory_reserved(device) / 2**30)
```

PyTorch kann Speicher zur Wiederverwendung reservieren. *Allocated*, *reserved* und die Gesamtbelegung aus `nvidia-smi` sind deshalb unterschiedliche Werte. Andere Programme und Treiber benötigen ebenfalls Speicher. Auf dem Spark prüfst du zusätzlich die systemweite Belegung des gemeinsamen Speichers.

Bei `K` Teilbatches pro Update und sonst gleichen festen Längen lautet die Tokenzahl `steps * B * T * K`. Bei Padding oder ignorierten Zielen musst du die tatsächlich bewerteten Positionen zählen. Unterscheide dabei die Modellrechnung auf allen Positionen vom Loss auf den nicht ignorierten Positionen.

### Von der Messung zu einer groben Laufzeit

Für ein geplantes **Tokenbudget**, also die gesamte Menge verarbeiteter Ziel-Tokens, gilt näherungsweise:

```text
Trainingszeit = geplante Ziel-Tokens / gemessene Ziel-Tokens pro Sekunde
```

Oder für eine feste Anzahl Updates: `Updates * Sekunden pro Update`.

**Reines Rechenbeispiel, keine Messung auf einem deiner Geräte:** Bei angenommenen 1000 Ziel-Tokens pro Sekunde würden 2 Millionen Ziel-Tokens ungefähr 2000 Sekunden, also 33 Minuten und 20 Sekunden reine gemessene Trainingszeit benötigen. Validierung, Speichern, Starten und Unterbrechungen kommen zusätzlich dazu. Bei doppelt so vielen Tokens wäre die Zeit unter denselben Bedingungen ungefähr doppelt so lang.

Bei zufällig gezogenen Textfenstern ist ein Tokenbudget keine Zusage, dass jeder Datensatz-Token gleich oft gesehen wurde. Eine **Epoche** bezeichnet dagegen normalerweise einen vollständigen Durchgang durch den festgelegten Datensatz. Notiere, welche Zählweise dein Programm verwendet.

Miss Validierungs- und Speicherintervalle ebenfalls oder plane eine sichtbar als Annahme gekennzeichnete Zeitreserve ein. Wiederhole die Abschätzung nach einer Änderung an Kontextlänge, Modellgrösse, Batch, Präzision oder Attention-Verfahren. Bei einem Laptop kann ein sehr kurzer Lauf das Verhalten unter länger andauernder Last schlecht abbilden; prüfe vor einer grossen Zeitplanung einen etwas längeren kleinen Lauf.

## Dein MINT-Projekt dokumentieren

Gemäss deiner Beschreibung ist eine **vorgängige Einreichung beim MINT-Team** vorgesehen. Kläre dort früh den vorgesehenen Ablauf und die konkreten Bedingungen. Die folgende Dokumentation ist ein praktischer Vorschlag; sie behauptet keine bestimmten schulischen Pflichtstunden oder Bewertungskriterien.

### Eine überschaubare erste Projektbeschreibung

Halte auf ungefähr einer Seite fest:

- **Vorhaben:** Ein kleines eigenes kausales Sprachmodell in Python und PyTorch entwickeln und verstehen.
- **Erste Etappe:** Einrichtung, Datenaufbereitung, einfacher Tokenizer, winziges Modell, kurze Lernversuche und dokumentierte Auswertung.
- **Eigene Arbeit:** Modellaufbau, Attention, Datenpipeline, später BPE, Tests und Interpretation.
- **Werkzeuge:** PyTorch für Tensoren, Autograd und Optimierung; weitere Bibliotheken mit ihren konkreten Aufgaben.
- **Offener Umfang:** Erweiterungen werden nach der ersten Etappe und anhand verfügbarer Zeit entschieden.
- **Mögliche Leitfrage:** Wie verändert sich die Vorhersagequalität, wenn mein Modell ausser dem letzten Token auch einen längeren Kontext nutzen kann?

Als spätere Forschungsfrage eignet sich beispielsweise der Vergleich von Byte-Tokenisierung und eigener BPE auf demselben Textmaterial. Damit der Vergleich verständlich bleibt, benennst du Unterschiede bei Vokabular, Parameterzahl, Kontext in Bytes und Trainingsbudget.

### Eine einfache Journalvorlage

Du kannst ein fortlaufendes `docs/project/journal.md` anlegen. Ein Eintrag pro Sitzung reicht. Kopiere die folgende Vorlage und fülle sie mit deinen eigenen Worten aus:

```markdown
## Datum und Sitzung

Datum:
Aktive Arbeitszeit ungefähr:
Zusätzliche reine Rechnerlaufzeit:

### Mein Ziel für heute

### Was ich gemacht habe

### Was ich verstanden oder neu gelernt habe

Ein Gedanke in meinen eigenen Worten:
Ein kleines Beispiel dazu:

### Was funktioniert hat

Prüfung oder Beobachtung:
Ergebnis:

### Probleme und meine Versuche

Fehlermeldung oder Beobachtung:
Meine Vermutung:
Was ich verändert habe:
Was danach passiert ist:

### Verwendete Hilfe

Quellen mit Link und verwendeter Stelle:
Bibliotheken und gegebenenfalls Versionen:
KI-Hilfsmittel und wofür ich sie eingesetzt habe:
Welchen Vorschlag ich selbst geprüft oder verändert habe:

### Nächster kleiner Schritt

### Offene Frage
```

Ein guter Eintrag darf kurz sein: «Ich dachte, `*` multipliziere Matrizen. Mein `(2,3)`-Beispiel zeigte, dass es elementweise rechnet. Mit `@` und `(3,4)` erhielt ich `(2,4)`. Ich kann den Unterschied jetzt an einem Beispiel erklären.» Das dokumentiert eine echte Erkenntnis.

KI-Unterstützung kannst du konkret beschreiben: «Eine KI erklärte mir die Maskierung. Ich schrieb die Maske selbst und prüfte, dass eine Änderung späterer Tokens die früheren Logits nicht verändert.» Trenne übernommene Vorschläge, eigene Umsetzung und deine Prüfung. Verwende die mit dem MINT-Team vereinbarten Regeln zur Angabe solcher Hilfe.

### Zu jedem Trainingslauf wenige feste Angaben speichern

Eine kleine Versuchsbeschreibung unter `experiments/` sollte enthalten: Datum, Fragestellung, Codezustand oder Git-Commit, Datenkennung und Aufteilung, Tokenizerkennung, Konfiguration und Parameterzahl, Seed, Gerät und Softwareversionen, verarbeitete Tokens, Zeit, Speicher, Loss-Werte sowie einige feste Textproben. Logs und grosse Checkpoints bleiben im lokalen Arbeitsordner; in der Beschreibung steht, wie du sie wiederfindest.

In `docs/project/decisions.md` hältst du Entscheidungen fest, zum Beispiel «BPE verschoben, damit ich zuerst den Trainingsablauf prüfen kann». In `research_questions.md` sammelst du Fragen. Ein nicht erfolgreicher Versuch gehört ebenfalls ins Journal, wenn du daraus etwas gelernt oder einen Fehler eingegrenzt hast.

## Dein konkreter Anfang

### Die ersten fünf Schritte

1. **Schreibe dein erstes Etappenziel auf:** «Ich möchte Text in Zahlen und zurück verwandeln und danach ein winziges Vorhersagemodell verstehen.» Notiere daneben, dass du die vorgängige Einreichung mit dem MINT-Team klärst.
2. **Prüfe Python und den Treiber:** Öffne eine neue PowerShell, prüfe die Befehle aus Phase 0A und notiere die Ergebnisse. Richte bei Bedarf normales 64-Bit-Python 3.12 über den offiziellen Weg ein.
3. **Lege die Umgebung ausserhalb von kDrive an:** Verwende den vorgeschlagenen lokalen Ordner, prüfe die Synchronisierungseinstellungen und kontrolliere `sys.executable`.
4. **Starte deinen ersten Textversuch:** Erzeuge selbst eine kleine `.py`-Datei im Ordner `notebooks/` und setze das Zeichen-Tokenizer-Beispiel aus Phase 3 mit `MINT!` um. Dafür benötigst du zunächst nur Python.
5. **Halte das Ergebnis fest und beginne PyTorch:** Schreibe deinen ersten Journaleintrag, installiere danach den geprüften passenden PyTorch-Build und führe den Rechentest aus. Auf der CPU kannst du auch dann weiterlernen, wenn die GPU noch nicht funktioniert.

### Deine erste Arbeitssitzung ausführlich

**Annahme:** Du hast etwa 60 bis 90 Minuten Zeit. Wenn eine Installation länger braucht, verschiebst du die restlichen Punkte. Der Zweck der Sitzung ist ein verständlicher kleiner Fortschritt.

**Minute 0 bis 10: Orientieren.** Öffne den Projektordner und schaue `README.md`, `notebooks/`, `src/mintclm/` und `docs/` an. Erkläre dir selbst: In `notebooks/` darf ich klein ausprobieren, in `src/` wachsen später wiederverwendbare Bausteine, in `tests/` prüfe ich Behauptungen und in `docs/` halte ich Erkenntnisse fest. Lege dein Journal an und notiere eine Frage, die dich interessiert.

**Minute 10 bis 30: Python prüfen und eine Umgebung anlegen.** Folge Phase 0A bis 0C. Kontrolliere den Interpreterpfad. Eine Umgebung ist erst dann sinnvoll eingerichtet, wenn du weisst, welches Python dein Editor und deine Konsole tatsächlich verwenden. Falls Python bereits gut funktioniert, brauchst du diese Zeit möglicherweise nicht vollständig.

**Minute 30 bis 50: Dein erstes Erfolgserlebnis bauen.** Lege `notebooks/01_text_und_tokens.py` an. Tippe das Beispiel mit `MINT!` aus Phase 3 selbst ab. Sage vor der Ausführung voraus, wie viele verschiedene Zeichen du hast und wie lang die ID-Liste sein wird. Führe die Datei in der eingerichteten Umgebung aus:

```powershell
python 'C:\Users\wunde\MintCLM\MintCLM 1.0\notebooks\01_text_und_tokens.py'
```

Prüfe, dass die zurückübersetzte Zeichenfolge `MINT!` lautet. Ändere den Text danach zu `MINT! MINT!` und baue die Tabelle erneut auf. Es kommt ein Leerzeichen ins Vokabular; die IDs können sich durch die sortierte Tabelle ändern. Das zeigt, warum du später eine einmal gewählte Tabelle speichern und zusammen mit dem Modell verwenden musst.

**Minute 50 bis 65: Aus Text eine Lernaufgabe machen.** Füge die Idee von `ids[:-1]` und `ids[1:]` hinzu, zunächst auch ohne PyTorch. Gib beide Listen und ihre decodierten Texte aus. Für `MINT!` sollst du `MINT` und `INT!` sehen. Erkläre dir laut: «An der Stelle des M soll das Modell das I vorhersagen.» Noch wird nichts gelernt; du hast aber die zentrale Lernaufgabe bereits korrekt vorbereitet.

**Minute 65 bis 80, falls Zeit bleibt: PyTorch ausprobieren.** Installiere die passende Variante aus Phase 0D und starte den Diagnosecode aus Phase 0E. Wenn die Installation länger dauert, ist dies dein nächster Termin. Mache nach erfolgreichem Import zunächst die kleine Gradientenübung aus Phase 1, statt sofort einen Transformer zu schreiben.

**Die letzten 10 Minuten: Festhalten.** Notiere Arbeitszeit, Ergebnis, eine eigene Erklärung und deinen nächsten Schritt. Wenn etwas scheiterte, speichere die konkrete Fehlermeldung und deinen kleinsten Versuch. Beende die Sitzung mit einer kleinen klaren nächsten Aufgabe, etwa «Ich möchte verstehen, weshalb `w.grad` im Quadrat-Beispiel 4 ist».

### Dein frühes Erfolgserlebnis

Dein erster sichtbarer Meilenstein ist: **Du kannst einen eigenen Satz in Zahlen verwandeln und exakt zurückholen.** Der nächste ist: **Du kannst zu jeder Eingabeposition das richtige nächste Zieltoken zeigen.** Danach folgt ein Parameter, der durch Gradienten besser wird. Jeder dieser Schritte ist ein echter Bestandteil deines späteren Sprachmodells.

### Fragen zum Selbstprüfen

Nach der ersten Sitzung:

1. Weshalb verarbeitet das Modell IDs statt direkt sichtbare Buchstaben?
2. Warum braucht die Decodierung dieselbe Tabelle wie die Codierung?
3. Was bedeuten Eingabe `MINT` und Ziel `INT!` an der ersten Position?
4. Wo liegen mein Quellcode und meine virtuelle Umgebung, und warum habe ich das so gewählt?
5. Welcher konkrete kleine Test zeigt mir, dass meine Umwandlung stimmt?

Nach den nächsten Etappen:

6. Was unterscheidet eine Token-ID von einem lernbaren Parameter?
7. Welche Achsen beschreibt `(B,T,V)`?
8. Warum darf Position 3 ihr eigenes Eingabetoken sehen, aber nicht das Token an Position 4?
9. Was tun `backward()` und `optimizer.step()` jeweils?
10. Warum beweist sinkender Trainings-Loss allein noch keine gute Vorhersage auf unbekannten Texten?
11. Welche Regeln lernt BPE, und weshalb dürfen diese nicht bei jeder Generierung neu gelernt werden?
12. Was brauche ich zusätzlich zu den Gewichten, um ein Training sinnvoll fortzusetzen?
13. Warum kann ich aus 128 GB Speicher allein keine Trainingsdauer berechnen?
14. Welches Ergebnis meines letzten Versuchs kann ich mit einem Beispiel und einer Prüfung selbst erklären?

Du musst die späteren Fragen heute noch nicht beantworten können. Verwende sie als Wegmarken: Wenn du einen Begriff mit deinem eigenen kleinen Beispiel erklären kannst, hast du einen tragfähigen nächsten Baustein für MintCLM gewonnen.
