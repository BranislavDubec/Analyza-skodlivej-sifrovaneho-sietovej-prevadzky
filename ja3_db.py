
import psycopg2

import csv

BLACKLISTS = "ja3_fingerprints.csv"
def conn() :
    return psycopg2.connect(
        host="localhost",
        database="blacklistdb",
        user="postgres",
        password="postgres"
    )
dbconn = conn()

def InsertDataToTable(table, vals):
    """
    Simplifies the insert query and executes it.

    Parameters:
        table(str): Identifies the table to insert to.
        col(str): Identifies the column of the table.
        vals(): The value(s) to insert. Data type depends on the the DB.
    """


    sql = "INSERT INTO {0}  VALUES (%s,%s,%s,%s)".format(table)
    blcursor.execute(sql, (vals))
    bldb.commit()
def GetDataFromCSV():
    links = tuple()
    with open(BLACKLISTS) as csvf:
        reader = csv.reader(csvf)
        for row in reader:
            links = tuple()
            if row[0].startswith('#'):
                continue
            for r in row:
                r = tuple((r,))
                links = links + r
            InsertDataToTable("ja3", links)
        links = [x for x in links if x!='']
        return links
try:
    # blacklist BD connection
    bldb = dbconn
    blcursor = dbconn.cursor()

except Exception as e:
    print(str(e))
sheet = GetDataFromCSV()

