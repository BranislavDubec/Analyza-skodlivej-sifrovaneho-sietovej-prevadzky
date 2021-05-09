# Poiadavky na spustenie

Je potrebné ma nainštalovanı programovací jazyk **Python**, a nainštalovanú **postgresSQL** databázovı systém.



# Python libraries

Pre  fungovanie skriptov je potrebné nainštalova nasledujúce kninice:
- os, hashlib, pandas, re, numpy, csv , time
- pyshark,  psycopg2,  textwrap,
- scikit-learn, scikitplot

# Pred spustením

Je potrebné zmeni pred spustením heslo pre uívate¾a **postgres** v psql, v linuxovıch systémoch napr.:
**sudo -u postgres psql -c 
"ALTER USER postgres PASSWORD 'postgres';"**
A vytvorenie databázy **blacklistdb**:
**CREATE DATABASE blacklistdb;**
Ïa¾ši krok je doplnenie hodnôt do databázy, v prieèinku **db_backup/** je vytvorenı backup databáze, ktorı sa spustí:
**psql -U postgres -p 5432 -h localhost -d blacklistdb < db_backup/ja3_backup.backup**
V tomto prípade u nie je potrebné spusti skript **ja3_db.py**, keïe je databáza u naplnená hodnotami.  Tento krok je moné nahradi tak, e sa vytvorí v databáze tabu¾ka s takouto schémou:
CREATE TABLE ja3 (
    ja3_md5 character(500) NOT NULL PRIMARY KEY,
    firstseen timestamp without time zone,
    lastseen timestamp without time zone,
    listingreason character varying(30)
);
a následne spusti skript **ja3_db.py**

# Vytvorenie datasetov
Skript **data_mining.py** vytvorí datasety z pcap súborov v prieèinku **pcap_used/**, a uloí ich ako csv súbory do prieèinku **csv_used/**. Takisto tieto dáta znormalizuje, a uloí tieto normalizované dáta do prieèinku **csv_used/normalized/**. Pre úèely prezentácie sú tieto súbory u vytvorené, take tento skript nie je potrebné spúša.

Spustením skriptu **ml.py** sa naèítajú vybrané súbory z prieèinku **csv_used/normalized/** a vykonajú sa techniky detekcie anomálií, viï komentáre v skripte.
