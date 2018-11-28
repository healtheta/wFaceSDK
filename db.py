import mysql.connector
from mysql.connector import errorcode
import numpy as np
import json

cursor = None
cnx = None


TABLES = {}
TABLES['encode_map'] = (
    "CREATE TABLE `encode_map` ("
    "  `enc_no` int(11) NOT NULL AUTO_INCREMENT,"
    "  `value` varchar(8196) NOT NULL,"
    "  `label` varchar(256) NOT NULL,"
    "  PRIMARY KEY (`enc_no`)"
    ") ENGINE=InnoDB")

def connect():
	try:
		global cursor
		global cnx
		cnx = mysql.connector.connect(user='', password='',
                              host='127.0.0.1',
                              database='wface')
		cursor = cnx.cursor()
		print(cursor)
	except mysql.connector.Error as err:
		if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
			print("Something is wrong with your user name or password")
			return False
		elif err.errno == errorcode.ER_BAD_DB_ERROR:
			print("Database does not exist")
			return False
		else:
			print(err)
			return False
			
	return True
		
							  
def close():
	if cursor is not None:
		cursor.close()
		
def create():
	global cursor
	print(cursor)
	if cursor is not None:
		try:
			cursor.execute(TABLES['encode_map'])
		except mysql.connector.Error as err:
			if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
				print("already exists.")
				return True
			else:
				print(err.msg)
				return False
				
		return True
	else:
		print("please connect to database")
		return False
		
		
add_encode = ("INSERT INTO encode_map "
               "(value, label) "
               "VALUES (%s, %s)")		
def add(label, value):
	global cursor
	global cnx
	if cursor is not None:
		try:
			data_encode = [value, label]
			cursor.execute(add_encode, data_encode)
			cnx.commit()
		except mysql.connector.Error as err:
			print(err)
			return False
			
	else:
		return False;

query = ("SELECT label, value FROM encode_map ")		
def list():
	global cursor
	global cnx
	if cursor is not None:
		cursor.execute(query)
		labels = []
		encodes = []
		for (label, value) in cursor:
			labels.append(label)
			y = json.loads(value)
			y = np.array(y)
			encodes.append(y)
		return [labels, encodes]
			
	else:
		return None
	