# MintCLM: Deine Modellreihe und dein Weg

**Stand: 15. November 2026.** Diese Seite beschreibt Ziele und offene Entscheidungen, keine bereits fertigen Modelle. Für Übungen und technische Grundlagen siehe den [ausführlichen Leitfaden](leitfaden.md). Die [Browser-Version dieser Übersicht](modellreihe.html) ist leichter zu lesen.

## Der Name

**MintCLM-1:1M** bedeutet: MintCLM, erste Modellfamilie, ungefähr eine Million lernbare Parameter. `M` steht für Million, `B` für Milliarde. Die Zahlen sind vorerst ungefähre Ziele; die tatsächliche Parameterzahl wird nach dem Bau gezählt.

| Name | Rolle | Stand |
|---|---|---|
| MintCLM-1:1M | Erstes selbst trainiertes Modell auf dem LOQ | Erstes Ziel |
| MintCLM-1:100M oder MintCLM-1:300M | Mögliche grössere Versionen | Offen; nicht beide nötig |
| MintCLM-1:4B | Langfristiges Ziel | Offen; Hardware und Trainingszeit müssen geprüft werden |

Der bestehende Ordner `MintCLM 1.0` enthält das ganze Projekt. Er ist weder das erste Modell noch eine Behauptung, dass es schon existiert. Unter Windows ist `:` in Dateinamen verboten. Als Dateiname eignet sich z. B. `mintclm-1-1m.yaml`, als Anzeigename weiterhin `MintCLM-1:1M`.

Die Familiennummer kann später eine grundlegend überarbeitete Modellfamilie kennzeichnen. Sie ist deine eigene Namensentscheidung, kein technischer Standard. Eine grössere Variante wird normalerweise mit eigener Architektur und neu initialisierten Gewichten trainiert: Der kleinere Trainingsstand wächst nicht automatisch mit. Es gibt Forschungsansätze zur Übertragung, aber sie sind nicht Teil der ersten Etappe.

## Der erste Meilenstein: MintCLM-1:1M

Du baust mit PyTorch einen kleinen Decoder-Transformer, trainierst ihn von zufällig initialisierten Gewichten auf geeigneten Texten und zeigst, dass er gelernt hat. Er soll einen Starttext fortsetzen, sich speichern und wieder laden lassen. Die genaue Konfiguration legen wir erst bei der Implementierung fest und zählen die Parameter; eine Million ist ein Zielwert, keine Garantie auf eine bestimmte Qualität.

Für eine Vorführung an deine Eltern:

1. Zeige dieselben Starttexte vor und nach dem Training.
2. Zeige den Verlauf des Loss auf Trainingsdaten und zurückgehaltenen Validierungsdaten.
3. Erkläre, welche Teile du selbst implementiert hast und welche Bibliotheken dir helfen.
4. Zeige auch Ausgaben, die nicht funktionieren. Ein kleineres Modell ist noch kein verlässlicher Chatbot.
5. Halte Gerät, Speicherbedarf, verarbeitete Tokens und gemessene Trainingsgeschwindigkeit fest.

Die vorhandenen Modell-, Trainings- und Testdateien sind momentan leer oder Platzhalter. Dieser Plan allein startet noch kein Training. Der [Leitfaden](leitfaden.md) erklärt die einzelnen Lernschritte.

## Geräte und Zugang

| Gerät | Geplante Rolle | Zugang |
|---|---|---|
| Lenovo LOQ, RTX 5060 | Entwicklung und erstes 1M-Training | Ausgangspunkt |
| DGX Spark, 128 GB gemeinsamer Speicher | Später grössere Versuche | Eltern nach einer gelungenen ersten Demo fragen |
| AI-Laptop, RTX PRO 5000, 24 GB VRAM und 128 GB RAM | Mögliche weitere Versuche | Voraussichtlich zugänglich; noch klären |
| H200 am Institut | Mögliche spätere grössere Versuche | Offen; Freigabe, Zeit und gemeinsame Nutzung klären |
| Möglicher Cluster mit sechs RTX PRO 6000 | Zukunftsmöglichkeit | Anschaffung und Zugang ungewiss |

Mehr Speicher bedeutet nicht automatisch schnelleres Training. Der Spark-Speicher wird geteilt. RAM im Laptop ist nicht dasselbe wie GPU-Speicher. Laufzeiten und machbare Grössen werden erst durch einen vollständigen Trainingsversuch auf dem jeweiligen Gerät bestimmt. Bei geteilter Hardware sind die Regeln der betreibenden Stelle zu beachten.

## CLM, GPT und Chat

CLM beschreibt das Lernziel: Aus bisherigem Text wird das nächste Token vorhergesagt. GPT bedeutet «Generative Pre-trained Transformer»: Ein GPT-artiges Modell kann ein kausales Decoder-Transformer-Sprachmodell sein. Du musst also nicht von CLM zu etwas völlig anderem wechseln. MintCLM kann schon in seiner ersten Transformer-Version GPT-artig sein, wenn es diese Eigenschaften erfüllt.

Ein **Basismodell** setzt zunächst vor allem Text fort. Für einen hilfreichen Chat-Assistenten braucht es meist weitere Arbeit, etwa passende Dialogdaten, zusätzliches Training, Tests und eine Oberfläche. «GPT-artig» garantiert weder eine bestimmte Modellgrösse noch ChatGPT-Qualität. Ein 1M-Modell bleibt sehr klein, auch wenn seine Architektur GPT-artig ist.

## Später aus Rückmeldungen lernen

Eine Unterhaltung allein verändert die Gewichte normalerweise nicht. Kontext oder gespeicherte Einstellungen können Antworten beeinflussen, ohne dass das Modell trainiert wird. Eine mögliche spätere Verbesserungsschleife wäre:

1. Nutzer können freiwillig Rückmeldungen oder geeignete Beispiele zur Verwendung freigeben.
2. Daten werden geprüft, bereinigt und nur für den vereinbarten Zweck verwendet; vertrauliche und persönliche Informationen werden geschützt.
3. Ausgewählte Beispiele werden als getrennte, versionierte Trainingsdaten vorbereitet; unabhängige Testdaten bleiben zurückgehalten.
4. Ein neuer Modellstand wird kontrolliert zusätzlich trainiert und mit dem bisherigen verglichen – auch darauf, ob frühere Fähigkeiten verloren gehen.
5. Nur wenn der neue Stand die gewünschten Prüfungen besteht, wird er eingesetzt. Der alte Stand bleibt für eine Rückkehr verfügbar.

Wenige gute Rückmeldungen können helfen, konkrete Fehler zu erkennen oder ein Modell gezielt anzupassen. Viele Nutzer sind nicht zwingend nötig. Für breit angelegte Verbesserungen ist oft viel sorgfältig ausgewähltes Material nötig; die Anzahl allein reicht nicht. Direkte automatische Updates aus jeder Unterhaltung sind kein frühes Projektziel: Daten können falsch, manipuliert oder persönlich sein und Training kann Fähigkeiten verschlechtern. Eine Sammlung oder Nutzung von Nutzerdaten ist derzeit weder eingerichtet noch gestartet.

## Nächster Schritt

Im [Leitfaden](leitfaden.md) mit Einrichtung und den Grundlagen beginnen und danach MintCLM-1:1M Schritt für Schritt implementieren. Vor grösseren Läufen Architektur, Daten, gespeicherte Zustände, Tests und GPU-Nutzung nachvollziehbar machen.
