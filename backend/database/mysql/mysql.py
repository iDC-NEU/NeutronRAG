'''
Author: lpz 1565561624@qq.com
Date: 2025-07-30 19:26:29
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-05 21:11:47
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/database/mysql/mysql.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import pymysql
import re

def flatten_to_str(value):
    if isinstance(value, list):
        # 展平嵌套列表为一维
        flat = []

        def _flatten(x):
            for i in x:
                if isinstance(i, list):
                    _flatten(i)
                else:
                    flat.append(i)

        _flatten(value)
        return str(flat)
    else:
        return str(value)  # 非列表也转为字符串

####history schema###   用于创建用户的历史测评记录表
HISTORY_TABLE_SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS `{table_name}` (
    `id` VARCHAR(64) PRIMARY KEY,
    `query` TEXT NOT NULL,
    `answer` TEXT,
    `type` ENUM('GREEN', 'YELLOW', 'RED') DEFAULT NULL,

    `vector_response` TEXT,
    `graph_response` TEXT,
    `hybrid_response` TEXT,

    `vector_retrieval_result` TEXT,
    `graph_retrieval_result` TEXT,

    `vector_evaluation` TEXT,
    `graph_evaluation` TEXT,
    `hybrid_evaluation` TEXT,

    `avg_vector_evaluation` TEXT,
    `avg_graph_evaluation` TEXT,
    `avg_hybrid_evaluation` TEXT,

    `v_error` TEXT,
    `g_error` TEXT,
    `h_error` TEXT,

    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


USER_INFO_TABLE_SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS `user_info` (
    `user_id` VARCHAR(64) PRIMARY KEY,
    `user_name` VARCHAR(20) NOT NULL,
    `table_num` INT DEFAULT 0
);
"""

class MySQLManager:
    def __init__(self, host="127.0.0.1", port=3307, user="root", password="a123456", database="chat"):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            autocommit=True
        )
        self.cursor = self.conn.cursor()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def is_valid_table_name(self, table_name):
        return re.match(r'^[A-Za-z0-9_-]+$', table_name) is not None

    def create_user_table(self, user_id, table_name):
        if not self.is_valid_table_name(table_name):
            raise ValueError("Invalid table name.")

        full_table_name = f"user{user_id}_{table_name}"
        schema_sql = f"""
        CREATE TABLE IF NOT EXISTS `{full_table_name}` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `question` TEXT NOT NULL,
            `answer` TEXT,
            `type` ENUM('GREEN', 'YELLOW', 'RED') DEFAULT 'GREEN',
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.cursor.execute(schema_sql)
        return full_table_name

    def table_exists(self, table_name):
        """
        检查当前数据库中某张表是否存在。
        :param table_name: 要检查的表名
        :return: True 如果存在，False 如果不存在
        """
        try:
            self.cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            """, (self.conn.db.decode(), table_name))
            result = self.cursor.fetchone()
            return result[0] > 0
        except Exception as e:
            print("❌ 检查表是否存在时出错：", str(e))
            raise

    # def create_user_info(self):
    #     """
    #     创建 user_info 表（如果尚不存在）
    #     """
    #     if self.table_exists("user_info"):
    #         print("ℹ️ user_info 表已存在，无需创建。")
    #         return

    #     try:
    #         self.cursor.execute(USER_INFO_TABLE_SCHEMA_TEMPLATE)
    #         print("✅ user_info 表已成功创建。")
    #     except Exception as e:
    #         print("❌ 创建 user_info 表时出错：", str(e))
    #         raise
    def get_database_table_summary(self):
        """
        获取当前数据库中所有表的数量及每张表的基本信息（字段数、创建时间等）。
        返回一个 dict，包含表名、字段数、创建时间等信息。
        """
        try:
            # 当前数据库名
            db_name = self.conn.db.decode()

            # 查询所有表
            self.cursor.execute("""
                SELECT table_name, create_time
                FROM information_schema.tables
                WHERE table_schema = %s
            """, (db_name,))
            tables = self.cursor.fetchall()

            summary = {
                "database": db_name,
                "total_tables": len(tables),
                "tables": []
            }

            for table_name, create_time in tables:
                # 查询字段数量
                self.cursor.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                """, (db_name, table_name))
                column_count = self.cursor.fetchone()[0]

                summary["tables"].append({
                    "table_name": table_name,
                    "column_count": column_count,
                    "create_time": str(create_time) if create_time else "unknown"
                })

            print(summary)

        except Exception as e:
            print("❌ 获取数据库信息失败：", str(e))
            raise

    def add_history_table(self, user_id, table_suffix):
        """
        创建一张新的历史记录表（如果不存在）。
        表名格式: user{user_id}_history_{table_suffix}
        同时将 user 表中对应用户的 table_num 字段 +1。
        """
        user_id = str(user_id)

        if not self.is_valid_table_name(table_suffix):
            raise ValueError("非法表名，仅支持字母、数字、下划线")

        full_table_name = f"user{user_id}_history_{table_suffix}"

        if self.table_exists(full_table_name):
            print(f"ℹ️ 表 `{full_table_name}` 已存在，无需创建。")
            return full_table_name

        try:
            # 创建历史表
            create_sql = HISTORY_TABLE_SCHEMA_TEMPLATE.format(table_name=full_table_name)
            self.cursor.execute(create_sql)
            print(f"✅ 成功创建历史表 `{full_table_name}`")

            # 更新 user 表中 table_num 字段 +1
            self.cursor.execute("""
                UPDATE `user` SET table_num = table_num + 1 WHERE id = %s
            """, (user_id,))
            print(f"🔄 已更新 `user` 表中 `{user_id}` 的 table_num +1")

            self.conn.commit()
            return full_table_name

        except Exception as e:
            self.conn.rollback()
            print(f"❌ 创建历史表 `{full_table_name}` 或更新 user 表失败: {str(e)}")
            raise

    def del_history_table(self, user_id, table_suffix):
        """
        删除指定用户的历史记录表，并同步减少 user 表中的 table_num 计数。
        表名格式: user{user_id}_history_{table_suffix}
        """
        if not self.is_valid_table_name(table_suffix):
            raise ValueError("非法表名，仅支持字母、数字、下划线")

        full_table_name = f"user{user_id}_history_{table_suffix}"

        if not self.table_exists(full_table_name):
            print(f"ℹ️ 表 `{full_table_name}` 不存在，无法删除。")
            return False

        try:
            # 删除表
            self.cursor.execute(f"DROP TABLE `{full_table_name}`")
            print(f"🗑️ 成功删除历史表 `{full_table_name}`")

            # 更新 user 表中的 table_num -= 1，确保不为负数
            self.cursor.execute(
                "UPDATE user SET table_num = GREATEST(table_num - 1, 0) WHERE id = %s",
                (user_id,)
            )
            print(f"📉 用户 {user_id} 的表计数 table_num -1")

            return True
        except Exception as e:
            print(f"❌ 删除表 `{full_table_name}` 失败: {str(e)}")
            raise



    
    # def add_user(self, user_id, user_name):
    #     """
    #     添加一个新用户到 user_info 表中。
    #     如果已存在则报错。
    #     """
    #     if not self.is_valid_table_name(user_id) or not self.is_valid_table_name(user_name):
    #         raise ValueError("非法 user_id 或 user_name。")

    #     # 检查是否已存在
    #     self.cursor.execute("SELECT COUNT(*) FROM user_info WHERE user_id = %s", (user_id,))
    #     if self.cursor.fetchone()[0] > 0:
    #         raise ValueError(f"用户 `{user_id}` 已存在。")

    #     try:
    #         self.cursor.execute(
    #             "INSERT INTO user_info (user_id, user_name, table_num) VALUES (%s, %s, %s)",
    #             (user_id, user_name, 0)
    #         )
    #         print(f"✅ 成功添加用户 `{user_id}`（{user_name}）")
    #     except Exception as e:
    #         print(f"❌ 添加用户失败: {str(e)}")
    #         raise

    def delete_user(self, user_id):
        """
        删除用户信息并删除其所有历史记录表（user{user_id}_history_*）。
        """
        if not self.is_valid_table_name(user_id):
            raise ValueError("非法 user_id")

        try:
            # 查找所有该用户创建的历史表
            self.cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %s AND table_name LIKE %s
            """, (self.conn.db.decode(), f"user{user_id}_history_%"))

            user_tables = [row[0] for row in self.cursor.fetchall()]

            # 删除所有表
            for table in user_tables:
                self.cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
                print(f"🗑️ 已删除表 `{table}`")

            # 删除 user_info 中的记录
            self.cursor.execute("DELETE FROM user WHERE user_id = %s", (user_id,))
            print(f"✅ 已删除用户 `{user_id}` 及其 {len(user_tables)} 张历史表")

        except Exception as e:
            print(f"❌ 删除用户失败: {str(e)}")
            raise



    def get_user_history_suffixes(self, user_id):
        """
        获取某个用户所有历史表的后缀名（如 testset1, testset2...）
        :param user_id: 用户ID（如 3）
        :return: List[str] 后缀名列表
        """
        if not self.is_valid_table_name(user_id):
            raise ValueError("非法 user_id")

        try:
            self.cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %s AND table_name LIKE %s
            """, (self.conn.db.decode(), f"user{user_id}_history_%"))

            table_names = [row[0] for row in self.cursor.fetchall()]

            # 提取后缀部分
            suffixes = []
            prefix = f"user{user_id}_history_"
            for table_name in table_names:
                if table_name.startswith(prefix):
                    suffixes.append(table_name[len(prefix):])

            return suffixes

        except Exception as e:
            print(f"❌ 获取用户 `{user_id}` 的历史表失败: {str(e)}")
            raise

    
    def get_user_history_table_count(self, user_id):
        """
        获取某个用户创建的历史记录表总数。
        表名格式为: user{user_id}_history_*

        :param user_id: 用户ID（字符串或数字）
        :return: int 表的数量
        """
        if not self.is_valid_table_name(user_id):
            raise ValueError("非法 user_id")

        try:
            self.cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_name LIKE %s
            """, (self.conn.db.decode(), f"user{user_id}_history_%"))

            count = self.cursor.fetchone()[0]
            return count

        except Exception as e:
            print(f"❌ 获取用户 `{user_id}` 历史表数量失败: {str(e)}")
            raise

    def print_table_contents(self, table_name, limit=None):
        """
        打印指定表中的所有数据（可选限制条数）
        :param table_name: 表名
        :param limit: 限制最多打印的条数（默认None表示不限制）
        """
        if not self.is_valid_table_name(table_name):
            raise ValueError("非法表名，仅允许字母、数字、下划线")

        try:
            # 查询字段名
            self.cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
            columns = [col[0] for col in self.cursor.fetchall()]

            # 查询所有行数据
            sql = f"SELECT * FROM `{table_name}`"
            if limit:
                sql += f" LIMIT {limit}"

            self.cursor.execute(sql)
            rows = self.cursor.fetchall()

            if not rows:
                print(f"ℹ️ 表 `{table_name}` 为空")
                return

            # 打印表头
            print(f"\n📋 表 `{table_name}` 内容：")
            print("-" * 80)
            print(" | ".join(columns))
            print("-" * 80)

            # 打印每行
            for row in rows:
                print(" | ".join(str(value) if value is not None else "NULL" for value in row))
            print("-" * 80)

        except Exception as e:
            print(f"❌ 打印表 `{table_name}` 内容失败: {str(e)}")
            raise


    def add_table_num_column_to_user(self):
        """
        向 `user` 表添加 `table_num` 列（如果不存在）。
        """
        try:
            # 检查该列是否已存在
            self.cursor.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
            """, (self.conn.db.decode(), "user", "table_num"))
            exists = self.cursor.fetchone()[0] > 0

            if exists:
                print("ℹ️ 列 `table_num` 已存在于 `user` 表中。")
                return

            # 添加列
            self.cursor.execute("""
                ALTER TABLE `user`
                ADD COLUMN `table_num` INT DEFAULT 0
            """)
            print("✅ 已成功为 `user` 表添加列 `table_num`。")

        except Exception as e:
            print(f"❌ 添加列失败: {str(e)}")
            raise


    def refresh_user_table_num(self):
        """
        刷新 user_info 表中所有用户的 table_num 字段。
        对每个用户重新统计其创建的历史表数量，并更新到 user_info 中。
        """
        try:
            # 获取所有用户的 user_id
            self.cursor.execute("SELECT id FROM user")
            users = [row[0] for row in self.cursor.fetchall()]

            for user_id in users:

                # 统计该用户拥有的历史表数
                self.cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = %s AND table_name LIKE %s
                """, (self.conn.db.decode(), f"user{user_id}_history_%"))
                table_count = self.cursor.fetchone()[0]

                # 更新 user_info 中该用户的 table_num 字段
                self.cursor.execute(
                    "UPDATE user SET table_num = %s WHERE id = %s",
                    (table_count, user_id)
                )
                print(f"✅ 已更新用户 `{user_id}` 的 table_num = {table_count}")

            self.conn.commit()

        except Exception as e:
            print(f"❌ 刷新 table_num 失败: {str(e)}")
            raise

    def modify_evaluation_columns_to_text(self, user_id, suffix):
        """
        将指定用户历史记录表中的多个 FLOAT 类型列修改为 TEXT 类型
        以支持存储完整 JSON 数据结构。
        """
        table_name = f"user{user_id}_history_{suffix}"
        columns_to_modify = [
            "vector_evaluation", "graph_evaluation", "hybrid_evaluation",
            "avg_vector_evaluation", "avg_graph_evaluation", "avg_hybrid_evaluation"
        ]

        try:
            for col in columns_to_modify:
                alter_sql = f"ALTER TABLE `{table_name}` MODIFY COLUMN `{col}` TEXT;"
                self.cursor.execute(alter_sql)
                print(f"✅ 修改列 `{col}` 为 TEXT 类型成功")
            print(f"🎉 表 `{table_name}` 所有指定列已更新为 TEXT")
        except Exception as e:
            print(f"❌ 修改表 `{table_name}` 时出错: {e}")
    
    def insert_record_to_history_table(self, user_id, table_suffix, record):
        """
        将一条记录插入指定用户的历史记录表中。
        表名格式: user{user_id}_history_{table_suffix}
        """
        if not self.is_valid_table_name(table_suffix):
            raise ValueError("非法表名，仅支持字母、数字、下划线")

        table_name = f"user{user_id}_history_{table_suffix}"

        insert_sql = f"""
        INSERT INTO `{table_name}` (
            id, query, answer, type,
            vector_response, graph_response, hybrid_response,
            vector_retrieval_result, graph_retrieval_result,
            vector_evaluation, graph_evaluation, hybrid_evaluation,
            avg_vector_evaluation, avg_graph_evaluation, avg_hybrid_evaluation,
            v_error, g_error, h_error
        ) VALUES (
            %(id)s, %(query)s, %(answer)s, %(type)s,
            %(vector_response)s, %(graph_response)s, %(hybrid_response)s,
            %(vector_retrieval_result)s, %(graph_retrieval_result)s,
            %(vector_evaluation)s, %(graph_evaluation)s, %(hybrid_evaluation)s,
            %(avg_vector_evaluation)s, %(avg_graph_evaluation)s, %(avg_hybrid_evaluation)s,
            %(v_error)s, %(g_error)s, %(h_error)s
        )
        """

        # 将非字符串类型字段（如 dict/list）转为 JSON 字符串
        def safe_json(value):
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return value

        # 构建参数字典
        params = {
            "id": record.get("id"),
            "query": record.get("query"),
            "answer": flatten_to_str(record.get("answer")),
            "type": record.get("type"),

            "vector_response": record.get("vector_response"),
            "graph_response": record.get("graph_response"),
            "hybrid_response": record.get("hybrid_response"),

            "vector_retrieval_result": safe_json(record.get("vector_retrieval_result")),
            "graph_retrieval_result": safe_json(record.get("graph_retrieval_result")),

            "vector_evaluation": safe_json(record.get("vector_evaluation")),
            "graph_evaluation": safe_json(record.get("graph_evaluation")),
            "hybrid_evaluation": safe_json(record.get("hybrid_evaluation")),

            "avg_vector_evaluation": safe_json(record.get("avg_vector_evaluation")),
            "avg_graph_evaluation": safe_json(record.get("avg_graph_evaluation")),
            "avg_hybrid_evaluation": safe_json(record.get("avg_hybrid_evaluation")),

            "v_error": record.get("v_error"),
            "g_error": record.get("g_error"),
            "h_error": record.get("h_error"),
        }

        try:
            self.cursor.execute(insert_sql, params)
            self.conn.commit()
            print(f"✅ 成功插入记录到 `{table_name}`")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ 插入记录失败: {str(e)}")
            raise
        


    def load_history_from_jsonl(self, jsonl_path, user_id, suffix):
        table_name = f"user{user_id}_history_{suffix}"

        columns = [
            "id", "query", "answer", "type",
            "vector_response", "graph_response", "hybrid_response",
            "vector_retrieval_result", "graph_retrieval_result",
            "vector_evaluation", "graph_evaluation", "hybrid_evaluation",
            "avg_vector_evaluation", "avg_graph_evaluation", "avg_hybrid_evaluation",
            "v_error", "g_error", "h_error"
        ]

        insert_sql = f"""
            INSERT INTO `{table_name}` ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
        """

        def safe_value(val):
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return val

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        values = [safe_value(data.get(col, None)) for col in columns]
                        self.cursor.execute(insert_sql, values)
                    except Exception as e:
                        print(f"⚠️ 第 {line_num} 行处理失败：{e}")

            print(f"✅ 成功导入数据到 `{table_name}`")
        except Exception as e:
            print(f"🚨 加载失败：{e}")


if __name__ == "__main__":
    db = MySQLManager(
        host="127.0.0.1",
        port=3307,
        user="root",
        password="a123456",
        database="chat"
    )
    # db.create_user_info()
    # db.get_database_table_summary()
    # # db.add_user(user_id="888",user_name="LPX")
    # db.add_history_table(user_id="10",table_suffix="rgb")
    # suffixes = db.get_user_history_suffixes(user_id='888')
    # print(suffixes)
    # count = db.get_user_history_table_count(user_id="888")
    # print(count)
    db.print_table_contents(table_name="user")
    db.refresh_user_table_num()
    # db.modify_evaluation_columns_to_text("10","rgb")
    # jsonl_path = "/home/lipz/NeutronRAG/NeutronRAG/backend/llmragenv/chat_history/rgb/000.json"
    # db.print_table_contents(table_name="user")
    # db.print_table_contents(table_name="10_history_rgb")
    # print(db.get_user_history_suffixes("10"))
    # db.del_history_table(user_id="10",table_suffix="rgb")
    # db.print_table_contents("user10_history_rgb")
    # db.load_history_from_jsonl(jsonl_path=jsonl_path,user_id="10",suffix="rgb")
    db.print_table_contents("user10_history_rgb")