# MintCLM

MintCLM ist dein Projekt zur Entwicklung und zum Training eines eigenen kausalen Sprachmodells in Python und PyTorch. Die erste Modellfamilie heisst **MintCLM-1**. Du beginnst auf dem Lenovo LOQ mit der RTX 5060 und möchtest später grössere Modelle untersuchen. Das Projekt enthält derzeit Struktur, Lernmaterial und Platzhalter: Modell, Training und Chat sind noch nicht implementiert.

## Deine Modellreihe

| Modellname | Bedeutung | Stand und Einsatz |
|---|---|---|
| **MintCLM-1:1M** | Familie 1, ungefähr 1 Million Parameter | Erstes Ziel: ein selbst trainiertes kleines Modell auf dem LOQ. |
| **MintCLM-1:100M** oder **MintCLM-1:300M** | Etwa 100 bzw. 300 Millionen Parameter | Mögliche spätere Zwischenschritte; Grösse und Hardware bleiben offen. |
| **MintCLM-1:4B** | Etwa 4 Milliarden Parameter | Langfristige Vision, noch kein zugesagtes Training oder garantierter Zugang zu Hardware. |

Die Parameterzahlen sind Zielgrössen, keine bereits vermessenen Modelle. `MintCLM 1.0` ist der bestehende Ordnername; er bezeichnet nicht das 1-Million-Parameter-Modell. Für Ordnernamen unter Windows verwendest du z. B. `MintCLM-1-1M`, weil `:` dort in Dateinamen nicht erlaubt ist.

Die [Projektübersicht im Browser](docs/project/modellreihe.html) erklärt den Plan, die Geräte, GPT und mögliche spätere Verbesserung durch Nutzerrückmeldungen. Die [Markdown-Fassung](docs/project/modellreihe.md) enthält dieselbe Planung.

Der ausführliche Projektleitfaden ist als [Browser-Version](docs/project/leitfaden.html) und [Markdown-Version](docs/project/leitfaden.md) verfügbar. Er führt dich mit Übungen und Tests zum ersten Modell; die späteren Ziele bleiben flexibel.

## Ziele

- Einen Textkorpus bereinigen und deduplizieren, einen BPE-Tokenizer implementieren und tokenisierte Daten mit `numpy.memmap` speichern.
- Eine Llama-inspirierte Architektur mit RMSNorm, RoPE, Grouped-Query Attention und SwiGLU entwickeln.
- Pre-Training mit AdamW, Cosine Learning Rate Decay, Gradient Clipping und Weights & Biases durchführen; Modellgrössen und Architekturentscheidungen durch Ablationen vergleichen.
- Supervised Fine-Tuning mit Loss-Masking, Evaluation und eine lokale Chat-Oberfläche ergänzen.
- Wo Hardware und Software es unterstützen, bfloat16 Mixed Precision und `torch.compile` nutzen.

## Projektstruktur

```text
MintCLM 1.0/
├── configs/               # Einstellungen für Training und Evaluation
│   ├── models/            # Geplante Konfigurationen für Modellgrössen
│   └── experiments/       # Geplante Einstellungen für einzelne Versuche
├── data/                  # Lokale Daten (nicht in Git)
│   ├── raw/               # Unveränderte Ausgangsdaten
│   ├── processed/         # Bereinigte und aufgeteilte Daten
│   ├── tokenized/         # Binäre, tokenisierte Daten
│   └── sft/               # Daten für Supervised Fine-Tuning
├── artifacts/             # Lokale, erzeugte Dateien (nicht in Git)
│   ├── tokenizer/         # Tokenizer-Vokabular und Merge-Regeln
│   ├── checkpoints/       # Modell- und Trainingsstände
│   └── outputs/           # Generierte Ausgaben und grosse Ergebnisse
├── src/mintclm/           # Python-Paket
│   ├── data/              # Datenaufbereitung und Datensätze
│   ├── tokenizer/         # BPE-Tokenizer
│   ├── model/             # Transformer-Bausteine
│   ├── training/          # Pre-Training, SFT und Lernratenplan
│   ├── evaluation/        # Auswertung und Kennzahlen
│   └── inference/         # Textgenerierung
├── scripts/               # Einstiegspunkte für die Projektphasen
├── app/                   # Lokale Chat-Oberfläche
├── tests/                 # Automatisierte Tests
├── experiments/           # Versuchsbeschreibungen und kleine Ergebnisse
│   ├── ablations/         # Fragestellungen und Vergleiche
│   └── results/           # Kleine, teilbare Ergebnisübersichten
├── docs/                  # Dokumentation und Projektnotizen
│   ├── data/              # Datenquellen und Aufbereitung
│   ├── architecture/      # Architekturentscheidungen
│   ├── training/          # Hardware- und Trainingseinrichtung
│   └── project/           # Forschungsfragen und Entscheidungen
├── notebooks/             # Experimente und Lernnotizen
├── requirements.txt       # Python-Abhängigkeiten
└── .gitignore             # Git-Ausschlüsse
```

Die Dateien in den Unterordnern sind Platzhalter für die spätere Arbeit. Dokumentiere bei jeder Datenquelle Herkunft und Nutzungsrechte. Halte Trainings-, Validierungs- und Testdaten getrennt; die konkrete Aufteilung und Deduplizierung werden später implementiert. Unter `experiments/` gehören nur kleine, zur Veröffentlichung geeignete Zusammenfassungen; grosse Messdaten und Checkpoints gehören in `artifacts/` oder ausserhalb des synchronisierten Ordners.

## Einrichtung

Eine aktuelle Python-Version (z. B. Python 3.11 oder 3.12), ein passender NVIDIA-Treiber und eine virtuelle Umgebung sind empfohlen. Installiere PyTorch **zuerst** gemäss dem [offiziellen Installationsassistenten](https://pytorch.org/get-started/locally/) für dein Betriebssystem, deine Architektur und eine zu deiner GPU passende CUDA-Version. Für Blackwell ist CUDA 12.8 oder neuer relevant; überprüfe insbesondere auf dem ARM-basierten DGX Spark die angebotenen Builds und NVIDIA-Hinweise. Ein lokal installiertes CUDA Toolkit ist für die üblichen PyTorch-Binaries nicht grundsätzlich erforderlich; der Treiber muss kompatibel sein.

Unter Windows PowerShell beispielsweise:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# PyTorch gemäss offiziellem Installationsassistenten installieren
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Unter Linux lauten die Befehle für die virtuelle Umgebung `python3 -m venv .venv` und `source .venv/bin/activate`. Die Abhängigkeiten sind bewusst nicht vollständig festgeschrieben; für reproduzierbare Experimente später die tatsächlich verwendeten Versionen dokumentieren. `torch.compile` und bfloat16 müssen auf jeder Zielplattform separat erprobt werden.

## Daten, Git und kDrive

`data/` und `artifacts/` sind für lokale, grosse Dateien vorgesehen; nur `.gitkeep`-Platzhalter werden durch Git erfasst. Checkpoints, binäre Arrays, virtuelle Umgebungen und `wandb/` sind zusätzlich ausgeschlossen. Keine Zugangsschlüssel oder fremde Daten ohne passende Nutzungsrechte einchecken.

**Wichtig:** `.gitignore` wirkt nur auf Git, nicht auf kDrive. Speichere grosse Datensätze, Umgebungen, Checkpoints und Logs am besten ausserhalb des synchronisierten kDrive-Ordners oder richte dort passende Synchronisierungsausschlüsse ein. Pfade dafür können später in den Konfigurationen hinterlegt werden. Bereits von Git verfolgte Dateien entfernt `.gitignore` nicht automatisch.
