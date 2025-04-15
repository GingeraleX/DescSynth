Analisi e Sintesi Audio con Python
Questo strumento analizza file audio (come input.wav) e crea nuove versioni manipolate.

Cosa ti Serve
Python: Se non lo hai, scaricalo da python.org.
File Audio: Un file audio chiamato input.wav (o altro nome, ma dovrai modificare lo script).

Installazione
Scarica i file: Metti audio_ftt.py, install_requirements.bat (Windows) o install_requirements.sh (macOS) e input.wav nella stessa cartella.
Installa le librerie:
Windows: Doppio click su install_requirements.bat.
macOS: Apri Terminale, vai alla cartella, scrivi chmod +x install_requirements.sh e poi ./install_requirements.sh.

Come Usare
Apri il terminale:
Windows: Cerca "cmd" e aprilo.
macOS: Apri "Terminale".
Vai alla cartella: Usa il comando cd [percorso della cartella] (es. cd Desktop).
Esegui lo script: Scrivi python audio_ftt.py e premi Invio.

Risultati
Vedrai informazioni sull'audio nel terminale.
Un nuovo file audio output_*.wav verrà creato nella stessa cartella.
Si aprirà una finestra con grafici dell'audio.

Modifiche
Puoi cambiare le impostazioni (come il file audio o il tipo di elaborazione) aprendo audio_ftt.py con un editor di testo e modificando le prime righe.

Note
Se hai problemi, assicurati di aver installato Python correttamente.
Se i grafici non appaiono, controlla l'installazione delle librerie.
Se vuoi cambiare file audio, modifica la linea AUDIO_FILE = 'input.wav' all'inizio dello script.