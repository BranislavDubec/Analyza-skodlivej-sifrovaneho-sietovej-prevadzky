#  Analyza skodlivej sifrovanej sietovej prevadzky

Pcap súbory, ktoré sú väčšie, ako je 100MB je možné stiahnuť odtiaľto: https://drive.google.com/file/d/18pciEGtNf6nf9UbwM6T8RRwGT50_mmiZ/view?usp=sharing

Skript ja3_db.py sa pripojí na vytvorenú postgres databázu blacklistdb, kde je vytvorená tabuľka ja3, a vloží hodnoty zo súboru ja3_fingerprints.csv.
Túto databázu s už vloženými hodnotami som backupol a je v súbore db_backup\ja3_backup.backup
Príkaz na vytvorenie databázy z backupu: pg_restore -h localhost -p 5432 -U postgres -d old_db -v db_backup\ja3_backup.backup

Skript data_mining.py analyzuje pcap súbory v priečinku pcap_used  a vytvorí csv súbory jednotlivých TCP tokov, a pomocou JA3 fingerprintov tiež vyhodnotí, či sú škodlivé alebo nie. Tieto csv súbory uloží do priečinku csv_used. Jednotlive csv súbory taktiež normalizuje a uloží do priečinku csv_used\normalized, kde sú jednotlivé csv súbory uložené bez hlavičky a indexu. Tieto priečinky obsahujú už vytvorené súbory z pcap súborov v pcap_used priečinku.

Skript ml.py sa stará o machine learning
 Bakalarska praca 
