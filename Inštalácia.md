# Po�iadavky na spustenie

Je potrebn� ma� nain�talovan� programovac� jazyk **Python**, a nain�talovan� **postgresSQL** datab�zov� syst�m.



# Python libraries

Pre  fungovanie skriptov je potrebn� nain�talova� nasleduj�ce kni�nice:
- os, hashlib, pandas, re, numpy, csv , time
- pyshark,  psycopg2,  textwrap,
- scikit-learn, scikitplot

# Pred spusten�m

Je potrebn� zmeni� pred spusten�m heslo pre u��vate�a **postgres** v psql, v linuxov�ch syst�moch napr.:
**sudo -u postgres psql -c 
"ALTER USER postgres PASSWORD 'postgres';"**
A vytvorenie datab�zy **blacklistdb**:
**CREATE DATABASE blacklistdb;**
�a��i krok je doplnenie hodn�t do datab�zy, v prie�inku **db_backup/** je vytvoren� backup datab�ze, ktor� sa spust�:
**psql -U postgres -p 5432 -h localhost -d blacklistdb < db_backup/ja3_backup.backup**
V tomto pr�pade u� nie je potrebn� spusti� skript **ja3_db.py**, ke�e je datab�za u� naplnen� hodnotami.  Tento krok je mo�n� nahradi� tak, �e sa vytvor� v datab�ze tabu�ka s takouto sch�mou:
CREATE TABLE ja3 (
    ja3_md5 character(500) NOT NULL PRIMARY KEY,
    firstseen timestamp without time zone,
    lastseen timestamp without time zone,
    listingreason character varying(30)
);
a n�sledne spusti� skript **ja3_db.py**

# Vytvorenie datasetov
Skript **data_mining.py** vytvor� datasety z pcap s�borov v prie�inku **pcap_used/**, a ulo�� ich ako csv s�bory do prie�inku **csv_used/**. Takisto tieto d�ta znormalizuje, a ulo�� tieto normalizovan� d�ta do prie�inku **csv_used/normalized/**. Pre ��ely prezent�cie s� tieto s�bory u� vytvoren�, tak�e tento skript nie je potrebn� sp���a�.

Spusten�m skriptu **ml.py** sa na��taj� vybran� s�bory z prie�inku **csv_used/normalized/** a vykonaj� sa techniky detekcie anom�li�, vi� koment�re v skripte.
