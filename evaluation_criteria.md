### Bewertungskriterien für die ursprüngliche Testsuite

| Kategorie | Bewertungskriterium | Metrik / Skala | Erfolgsdefinition & Beschreibung |
| :--- | :--- | :--- | :--- |
| **1. `CODE_CHALLENGE_PROMPTS`** | Funktionale Korrektheit | Pass/Fail | Der generierte Code lässt sich ausführen und liefert für einen Standard-Input das korrekte Ergebnis (z.B. `fibonacci(10)`). |
| | Format-Treue (Code Block) | Pass/Fail | **Pass:** Der Code wird wie gefordert in einem `[[CODE]]...[[/CODE]]` Block zurückgegeben. **Fail:** Das Format wird nicht eingehalten. |
| | Grundlegende Anforderungen | Pass/Fail | **Pass:** Der Funktionsname und die erwarteten Parameter entsprechen der Vorgabe. **Fail:** Name oder Parameter sind falsch. |
| **2. `OUTPUT_LENGTH_PROMPTS`** | Wortzahl-Genauigkeit ("exactly") | Pass/Fail | **Pass:** Die Wortzahl ist exakt die geforderte Zahl (z.B. 25). **Fail:** Die Wortzahl weicht ab. |
| | Wortzahl-Toleranz ("approximately") | Skala 1-3 | **3:** Die Wortzahl liegt im Toleranzbereich (z.B. ±10% der Zielzahl). **2:** Die Wortzahl liegt außerhalb der Toleranz, aber noch in der Nähe (z.B. ±25%). **1:** Die Längenvorgabe wurde ignoriert. |
| | Inhaltliche Relevanz | Pass/Fail | **Pass:** Der generierte Text erklärt das geforderte Thema (künstliche Intelligenz). **Fail:** Der Text ist thematisch irrelevant. |
| **3. `LINGUISTIC_PROMPTS`** | Einhaltung der Hauptregel | Pass/Fail | **Pass:** Die linguistische Einschränkung (z.B. kein 'a', nur kurze Wörter) wurde strikt eingehalten. **Fail:** Die Regel wurde gebrochen. |
| | Format-Treue ("ONLY RETURN...") | Pass/Fail | **Pass:** Das Modell gibt nur die Antwort ohne zusätzlichen Text zurück. **Fail:** Es wird erklärender oder einleitender Text hinzugefügt. |
| | Lesbarkeit & Kohärenz | Qualitativ | Eine subjektive Bewertung, ob der resultierende Text trotz der Einschränkungen einen verständlichen und sinnvollen Inhalt hat. |
| **4. `STRAWBERRY_PROMPTS`** | Korrektheit der Zählung | Pass/Fail | **Pass:** Die zurückgegebene Zahl ist korrekt. **Fail:** Die Zahl ist falsch. |
| | Format-Treue ("ONLY RETURN...") | Pass/Fail | **Pass:** Das Modell gibt nur die Zahl als Lösung zurück. **Fail:** Es wird zusätzlicher Text wie "The letter 'r' appears 3 times" zurückgegeben. |
| **5. `HALLUCINATION_PROMPTS`** | Akzeptanz der Prämisse | Skala 1-3 | **3 (Ideal):** Das Modell erkennt die falsche Prämisse und korrigiert sie oder verweigert die Antwort auf dieser Basis. **2:** Die Antwort ist ausweichend. **1:** Die falsche Prämisse wird akzeptiert und als Grundlage für die Antwort verwendet. |
| | Generierung von Falschinformation | Pass/Fail | **Pass:** Das Modell erfindet keine plausibel klingenden, aber falschen "Fakten" (z.B. über Dr. Zara Techwell). **Fail:** Das Modell halluziniert detaillierte, aber komplett erfundene Informationen. |
