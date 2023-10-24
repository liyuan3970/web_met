import pymysql


class Connector:
    def __init__(self, host, port, user, password, db):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.conn = None

    def connect(self):
        self.conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.db,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def execute_query(self, query, params=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result

    def execute_update(self, query, params=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)

    def execute_update_many(self, query, params=None):
        with self.conn.cursor() as cursor:
            cursor.executemany(query, params)

    def close(self):
        self.conn.close()
