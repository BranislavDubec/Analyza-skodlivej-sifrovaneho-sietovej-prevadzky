"""
    Script adds values to ja3 table from the 'ja3_fingerprints.csv' file.
"""
__author__ = "Branislav Dubec"
__credits__ = ["Petr Chmelar"]
__version__ = "1.0.0"

import psycopg2
import csv


BLACKLISTS = "ja3_fingerprints.csv"


def conn():
    return psycopg2.connect(
        host="localhost",
        database="blacklistdb",
        user="postgres",
        password="postgres"
    )


def insertDataToTable(table, vals):
    """
    Simplifies the insert query and executes it.
    Parameters:
        table(str): Identifies the table to insert to.
        vals(): The value(s) to insert. Data type depends on the the DB.
    """
    try:
        sql = "INSERT INTO {0}  VALUES (%s,%s,%s,%s)".format(table)
        blcursor.execute(sql, vals)
        dbconn.commit()
    except Exception as e:
        print(e)

"""
   Reads data from BLACKLISTS file and calls insertDataToTable for each row.
"""
def getDataFromCSV():
    with open(BLACKLISTS) as csvf:
        reader = csv.reader(csvf)
        for row in reader:
            links = tuple()
            if row[0].startswith('#'):
                continue
            for r in row:
                r = tuple((r,))
                links = links + r
            insertDataToTable("ja3", links)



dbconn = conn()
try:
    # blacklist BD connection
    blcursor = dbconn.cursor()

except Exception as e:
    print(str(e))
getDataFromCSV()
