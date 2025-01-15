# app/db.py

import psycopg2

def connect_db():
    return psycopg2.connect(
        dbname="plant_index",
        user="postgres",
        password="admin",
        host="192.168.1.49",
        port="5432"
    )
