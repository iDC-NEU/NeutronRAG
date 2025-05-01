import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from db_setup import db

class User(db.Model):
    """用户模型，映射到 'user' 数据表。"""
    id = db.Column(db.Integer, primary_key=True) # 主键 ID
    username = db.Column(db.String(50), unique=True, nullable=False) # 用户名 (唯一, 非空)
    email = db.Column(db.String(100), unique=True, nullable=False) # 邮箱 (唯一, 非空)
    phone = db.Column(db.String(15), unique=True, nullable=False) # 手机号 (唯一, 非空)
    password_hash = db.Column(db.String(255), nullable=False) # 存储密码哈希 (非空)
    create_time = db.Column(db.DateTime, default=datetime.datetime.utcnow) # 创建时间

    # 关系：一个用户可以有多个会话
    # cascade: 删除用户时，其会话也将被删除
    sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """设置密码（存储哈希值）。"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """检查密码是否匹配。"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class ChatSession(db.Model):
    """聊天会话模型，映射到 'chat_session' 数据表。"""
    id = db.Column(db.Integer, primary_key=True) # 会话 ID (主键)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # 外键，关联用户 ID (非空)
    session_name = db.Column(db.String(100), nullable=False, default=lambda: f"会话 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") # 会话名称 (非空)
    create_time = db.Column(db.DateTime, default=datetime.datetime.utcnow) # 创建时间

    # 关系：一个会话包含多条消息
    # order_by: 按时间戳升序排列消息
    # cascade: 删除会话时，其消息也将被删除
    messages = db.relationship('ChatMessage', backref='session', lazy=True, order_by='ChatMessage.timestamp', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<ChatSession {self.id} - {self.session_name}>'

class ChatMessage(db.Model):
    """聊天消息模型，映射到 'chat_message' 数据表。"""
    id = db.Column(db.Integer, primary_key=True) # 消息 ID (主键)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False) # 外键，关联会话 ID (非空)
    query = db.Column(db.Text, nullable=False) # 用户查询 (非空)
    vector_response = db.Column(db.Text, nullable=True) # Vector RAG 回答
    graph_response = db.Column(db.Text, nullable=True) # Graph RAG 回答
    hybrid_response = db.Column(db.Text, nullable=True) # Hybrid RAG 回答

    # 存储检索结果
    vector_retrieval_json = db.Column(db.Text, nullable=True) # 向量检索结果 (JSON 字符串)
    graph_retrieval_raw = db.Column(db.Text, nullable=True) # 原始图谱关系字符串列表 (JSON 字符串)

    rag_mode_used = db.Column(db.String(20), nullable=True) # 使用的 RAG 模式
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, index=True) # 时间戳 (带索引)

    def __repr__(self):
        return f'<ChatMessage {self.id} in Session {self.session_id}>'

    def to_dict(self):
        """将消息对象转换为字典 (用于 API 响应)。"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query": self.query,
            "vector_response": self.vector_response,
            "graph_response": self.graph_response,
            "hybrid_response": self.hybrid_response,
            "rag_mode_used": self.rag_mode_used,
            "timestamp": self.timestamp.isoformat() + 'Z'
        }