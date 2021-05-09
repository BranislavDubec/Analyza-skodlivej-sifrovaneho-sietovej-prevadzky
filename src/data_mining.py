"""
    Script create csv files from pcap files in folder pcap_used/*.pcap.
    These csv files contains information about TCP streams and their JA3 fingerprint.
    It determines whether is stream malicious based on comparing individual JA3 fingerprint with
    blacklisted values of JA3 fingerprints in postgres table 'ja3'.
    CSV files are in csv_used.
    Also creates normalized csv files, stored in csv_used/normalized.

"""
__author__ = "Branislav Dubec"
__version__ = "1.2.1"
__copyright__ = "Adel 0x4d31 Karimi"

# Copyright (c) 2019, Adel "0x4d31" Karimi.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import pyshark
import os
from hashlib import md5
import psycopg2
import pandas as pd
import re
from sklearn import preprocessing
import numpy as np


df = pd.DataFrame(columns=['duration', 'srcIp', 'srcPort', 'dstIp',
                           'dstPort', 'service', 'srcBytes', 'dstBytes', 'flag',
                           'land', 'urgent', 'ja3', 'ja3Ver', 'ja3Cipher',
                           'ja3Extension', 'ja3Ec', 'ja3Ecpf', 'blacklisted'])
# GREASE_TABLE Ref: https://tools.ietf.org/html/draft-davidben-tls-grease-00
GREASE_TABLE = {0x0a0a: True, 0x1a1a: True, 0x2a2a: True, 0x3a3a: True,
                0x4a4a: True, 0x5a5a: True, 0x6a6a: True, 0x7a7a: True,
                0x8a8a: True, 0x9a9a: True, 0xaaaa: True, 0xbaba: True,
                0xcaca: True, 0xdada: True, 0xeaea: True, 0xfafa: True}

ja3Blacklist = []

# connects to database to take malicious ja3 fingerprints
def conn():
    return psycopg2.connect(
        host="localhost",
        database="blacklistdb",
        user="postgres",
        password="postgres"
    )


dbconn = conn()

# returns all rows from ja3 table from database blacklistdb
def getDataFromTable():
    sql = "SELECT * FROM ja3"
    blcursor.execute(sql)
    _records = blcursor.fetchall()
    _records = [i[0] for i in _records]
    return _records


try:
    # blacklist BD connection
    blcursor = dbconn.cursor()

except Exception as e:
    print(str(e))

records = getDataFromTable()


# function returns fields, that ja3 uses to create fingerprint
def client_ja3(packet):
    # ja3 fields
    tls_version = ciphers = extensions = elliptic_curve = ec_pointformat = ""
    if 'handshake_version' in packet.tls.field_names:
        tls_version = int(packet.tls.handshake_version, 16)
        tls_version = str(tls_version)
    if 'handshake_ciphersuites' in packet.tls.field_names:
        p = str(packet.tls).split('\r\n')
        cipher_list = []
        ciphers = ''
        for sentence in p:
            if sentence.startswith('\tCipher Spec:') or sentence.startswith('\tCipher Suite:'):
                cipher = sentence.split('(')[1]
                cipher = cipher[:-1]
                if cipher not in GREASE_TABLE:
                    cipher = int(cipher, 0)
                    cipher_list.append(int(cipher))

        for num in cipher_list:
            ciphers = ciphers + str(num) + '-'
        ciphers = ciphers[:-1]

    if 'handshake_extension_type' in packet.tls.field_names:
        p = str(packet.tls).split('\r\n')
        extension_list = []
        extensions = ''
        for sentence in p:
            if sentence.startswith('\tType:'):
                extension = sentence.split('(')[1]
                extension = extension[:-1]
                if int(extension, 16) not in GREASE_TABLE:
                    extension = int(extension, 0)
                    extension_list.append(int(extension))

        for ext in extension_list:
            extensions = extensions + str(ext) + '-'
        extensions = extensions[:-1]
    if 'handshake_extensions_supported_group' in packet.tls.field_names:
        p = str(packet.tls).split('\r\n')
        ec_list = []
        elliptic_curve = ''
        for sentence in p:
            if sentence.startswith('\tSupported Group:'):
                eliptic = sentence.split('(')[1]
                eliptic = eliptic[:-1]
                if int(eliptic, 16) not in GREASE_TABLE:
                    eliptic = int(eliptic, 0)
                    ec_list.append(int(eliptic))

        for num in ec_list:
            elliptic_curve = elliptic_curve + str(num) + '-'
        elliptic_curve = elliptic_curve[:-1]
    if 'handshake_extensions_ec_point_format' in packet.tls.field_names:
        p = str(packet.tls).split('\r\n')
        ecpf_list = []
        ec_pointformat = ''
        for sentence in p:
            if sentence.startswith('\tEC point format:'):
                pointformat = sentence.split('(')[1]
                pointformat = pointformat[:-1]
                if int(pointformat, 16) not in GREASE_TABLE:
                    pointformat = int(pointformat, 0)
                    ecpf_list.append(int(pointformat))

        for num in ecpf_list:
            ec_pointformat = ec_pointformat + str(num) + '-'
        ec_pointformat = ec_pointformat[:-1]
    server_name = ""
    if 'handshake_extensions_server_name' in packet.tls.field_names:
        server_name = packet.tls.handshake_extensions_server_name
    # Create ja3
    ja3_string = ','.join([
        tls_version, ciphers, extensions, elliptic_curve, ec_pointformat])
    ja3 = md5(ja3_string.encode()).hexdigest()
    record = {
        "tcpStream": packet.tcp.stream,
        "serverName": server_name,
        "ja3": ja3,
        "ja3Algorithms": ja3_string,
        "ja3Version": tls_version,
        "ja3Ciphers": ciphers,
        "ja3Extensions": extensions,
        "ja3Ec": elliptic_curve,
        "ja3Ecpf": ec_pointformat
    }
    return record

# add or delete values from ja3 fields, so all the ja3 values have same number of features
def updateJA3(dic):
    while dic['ja3Ciphers'].count('-') < 35:
        if len(dic['ja3Ciphers']) == 0:
            dic['ja3Ciphers'] = dic['ja3Ciphers'] + str(0)
            continue
        dic['ja3Ciphers'] = dic['ja3Ciphers'] + '-' + str(0)
    while dic["ja3Extensions"].count('-') < 25:
        if len(dic['ja3Extensions']) == 0:
            dic['ja3Extensions'] = dic['ja3Extensions'] + str(0)
            continue
        dic['ja3Extensions'] = dic['ja3Extensions'] + '-' + str(0)
    while dic["ja3Ec"].count('-') < 5:
        if len(dic['ja3Ec']) == 0:
            dic['ja3Ec'] = dic['ja3Ec'] + str(0)
            continue
        dic['ja3Ec'] = dic['ja3Ec'] + '-' + str(0)
    if dic['ja3Ecpf'].count('-') < 1:
        if len(dic['ja3Ecpf']) == 0:
            dic['ja3Ecpf'] = dic['ja3Ecpf'] + str(0) + '-' + str(0)
        else:
            dic['ja3Ecpf'] = dic['ja3Ecpf'] + '-' + str(0)
    if dic['ja3Ciphers'].count('-') > 35:
        pos = [m.start() for m in re.finditer(r"-", dic['ja3Ciphers'])][35]
        dic['ja3Ciphers'] = dic['ja3Ciphers'][:pos]
    if dic["ja3Extensions"].count('-') > 25:
        pos = [m.start() for m in re.finditer(r"-", dic['ja3Extensions'])][25]
        dic["ja3Extensions"] = dic["ja3Extensions"][:pos]
    if dic["ja3Ec"].count('-') > 5:
        pos = [m.start() for m in re.finditer(r"-", dic['ja3Ec'])][5]
        dic["ja3Ec"] = dic["ja3Ec"][:pos]
    if dic['ja3Ecpf'].count('-') > 1:
        pos = [m.start() for m in re.finditer(r"-", dic['ja3Ecpf'])][1]
        dic["ja3Ecpf"] = dic["ja3Ecpf"][:pos]
    return dic

# add new tcp stream into dataframe
def processFirstPacket(packet):
    global df

    if packet.tcp.srcport == packet.tcp.dstport:
        land = 1
    else:
        land = 0
    try:
        srcIp = packet.ip.src
    except:
        srcIp = packet.ip.addr

    data = {'duration': 0,
            'srcIp': srcIp, 'srcPort': packet.tcp.srcport,
            'dstIp': packet.ip.dst, 'dstPort': packet.tcp.dstport,
            'service': packet.ip.proto, 'srcBytes': packet.length,
            'dstBytes': 0, 'flag': packet.tcp.flags_reset,
            'land': land, 'urgent': packet.tcp.flags_urg,
            'ja3': 0, 'ja3Ver': 0, 'ja3Cipher': [],
            'ja3Extension': [], 'ja3Ec': [], 'ja3Ecpf': [],
            'blacklisted': 0
            }
    df = df.append(data, ignore_index=True)

# updates TCP stream in dataframe
def processPacket(packet):
    global df
    id = int(packet.tcp.stream)
    isClientSrc = False
    try:
        srcIp = packet.ip.src
    except:
        srcIp = packet.ip.addr
    if srcIp == df.at[id, 'srcIp']:
        isClientSrc = True

    if isClientSrc:
        df.at[id, 'srcBytes'] = int(df.at[id, 'srcBytes']) + int(packet.length)
    else:
        df.at[id, 'dstBytes'] = int(df.at[id, 'dstBytes']) + int(packet.length)

    df.at[id, 'flag'] = int(df.at[id, 'flag']) + int(packet.tcp.flags_reset)
    df.at[id, 'urgent'] = int(df.at[id, 'urgent']) + int(packet.tcp.flags_urg)
    df.at[id, 'duration'] = packet.tcp.time_relative


columns = ['duration','srcPort','dstPort','service','srcBytes', 'dstBytes',
           'flag', 'land', 'urgent', 'ja3Ver']

columns.extend(['ja3Cipher' + str(i) for i in range(36)])
columns.extend(['ja3Extension' + str(i) for i in range(26)])
columns.extend(['ja3Ec' + str(i) for i in range(6)])
columns.extend(['ja3Ecpf' + str(i) for i in range(2)])
columns.extend(['blacklisted'])

# each value in each ja3 field is separated and becomes individual features in dataset
def separataJA3fields(dataf):
    dataf.drop(dataf[dataf['ja3']  == 0].index , inplace=True)
    dataf = dataf.drop('srcIp' , axis=1)
    dataf = dataf.drop('dstIp', axis=1)
    data = []
    ct_b = 0
    ct_g = 0
    new_df = pd.DataFrame(columns = columns, dtype=np.float64)
    for index, row in dataf.iterrows():
        data.append(float(row['duration']))
        data.append(float(row['srcPort']))
        data.append(float(row['dstPort']))
        data.append(float(row['service']))
        data.append(float(row['srcBytes']))
        data.append(float(row['dstBytes']))
        data.append(float(row['flag']))
        data.append(float(row['land']))
        data.append(float(row['urgent']))
        data.append(float(row['ja3Ver']))
        ciphers = row['ja3Cipher'].split('-')
        for cipher in ciphers:
            data.append(float(cipher))
        extensions = row['ja3Extension'].split('-')
        for extension in extensions:
            data.append(float(extension))
        ecs = row['ja3Ec'].split('-')
        for ec in ecs:
            data.append(float(ec))
        ecpfs = row['ja3Ecpf'].split('-')
        for ecpf in ecpfs:
            data.append(float(ecpf))
        data.append(float(row['blacklisted']))
        if int(row['blacklisted']) == 0:
            ct_b = ct_b + 1
        else:
            ct_g = ct_g + 1
        new_df.loc[len(new_df)] = data
        data = []
    print("Good: ",ct_b, "malicious: ",ct_g)
    return new_df

# create csv file of each TCP stream in \csv_used and normalized csv file in \csv_used\normalized of each TCP stream with JA3 fingerprint
def createCSVfromPcap(pcap, filename):
    global records, df
    session = []
    cap = pyshark.FileCapture(pcap, keep_packets=False)
    ct_h = 0
    ct=0
    for packet in cap:
        ct = ct + 1
        print("Packet number", str(ct), str(filename))
        try:
            if int(packet.tcp.stream) not in session:
                processFirstPacket(packet)
                session.append(int(packet.tcp.stream))
            else:
                processPacket(packet)
        except:
            pass
        try:
            if 'Client Hello' in packet.tls.record:
                ja3_dic = client_ja3(packet)
                ct_h = ct_h + 1

                for record in records:
                    if (ja3_dic['ja3'] in record):
                        df.at[int(ja3_dic['tcpStream']), 'blacklisted'] = 1
                df.at[int(ja3_dic['tcpStream']), 'ja3'] = ja3_dic['ja3']
                ja3_dic = updateJA3(ja3_dic)
                df.at[int(ja3_dic['tcpStream']), 'ja3Ver'] = ja3_dic['ja3Version']
                df.at[int(ja3_dic['tcpStream']), 'ja3Cipher'] = ja3_dic['ja3Ciphers']
                df.at[int(ja3_dic['tcpStream']), 'ja3Extension'] = ja3_dic['ja3Extensions']
                df.at[int(ja3_dic['tcpStream']), 'ja3Ec'] = ja3_dic['ja3Ec']
                df.at[int(ja3_dic['tcpStream']), 'ja3Ecpf'] = ja3_dic['ja3Ecpf']

        except:

            pass
    df.to_csv("csv_used" + '/' + str(filename)[:-5]  + '.csv')
    df = separataJA3fields(df)

    #normalize values in dataset
    df_normalize = df.loc[:, df.columns != 'blacklisted']
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_normalize.values)
    x_scaled = np.append(x_scaled, df.loc[:, df.columns == 'blacklisted'], axis=1)

    df = pd.DataFrame(x_scaled, columns=columns)
    df.to_csv("csv_used/normalized/"  + str(filename)[:-5] + '_normalized' + '.csv', header=False, index=False)
    print(filename)
    df = pd.DataFrame(columns=['duration', 'srcIp', 'srcPort', 'dstIp',
                               'dstPort', 'service', 'srcBytes', 'dstBytes', 'flag',
                               'land', 'urgent', 'ja3', 'ja3Ver', 'ja3Cipher',
                               'ja3Extension', 'ja3Ec', 'ja3Ecpf', 'blacklisted'])
    df = df.iloc[0:0]


for root, dirs, files in os.walk('pcap_used'):
    for name in files:
        filepath = root + os.sep + name
        if filepath.endswith("pcap"):
            createCSVfromPcap(filepath, name)

