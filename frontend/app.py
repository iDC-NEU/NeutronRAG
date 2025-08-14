import functools
import json
import os
import random
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

from typing import Union, Optional

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from werkzeug.security import generate_password_hash, check_password_hash
from zhipuai import ZhipuAI

#from config.config import Config
from db_setup import db
from models import User, ChatSession, ChatMessage
from database.mysql.mysql import MySQLManager
#from user import User

# --- RAG 核心逻辑导入 ---
RAG_CORE_LOADED = False
try:
    # 动态添加 backend 目录到 sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设 frontend 文件夹和 backend 在同一级目录
    backend_dir = os.path.join(os.path.dirname(current_dir), 'backend')
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)
    
    from llmragenv.demo_chat import Demo_chat
    from evaluator import simulate
    RAG_CORE_LOADED = True
except ImportError as e:
    print(f"警告：无法导入 RAG 核心逻辑：{e}")
    Demo_chat = None
    simulate = None

################字典存入的mysql是TEXT这个在还原回来#####################
def parse_json_field(value):
    try:
        return json.loads(value) if value else None
    except Exception as e:
        print(f"⚠️ 解析失败: {e}")
        return None


# =========================================
# 初始化与配置
# =========================================
load_dotenv()

app = Flask(__name__, instance_relative_config=True)

# 初始化自定义的 MySQL 管理器
try:
    # 保留从环境变量加载数据库配置的写法
    mysql = MySQLManager(
        host=os.environ.get('DB_HOST', '127.0.0.1'),
        port=int(os.environ.get('DB_PORT', 3307)),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', 'a123456'),
        database=os.environ.get('DB_NAME', 'chat')
    )
    print("MySQLManager 连接成功。")
except Exception as e:
    print(f"错误：MySQLManager 连接失败: {e}")
    mysql = None

# --- Flask 应用配置 ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-very-secret-key-that-you-should-change')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db_uri = os.environ.get('DATABASE_URL')
if not db_uri:
    print("错误：环境变量 DATABASE_URL 未设置！")
    DB_INIT_SUCCESS = False
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    DB_INIT_SUCCESS = True

# --- 初始化 SQLAlchemy ---
if DB_INIT_SUCCESS:
    try:
        db.init_app(app)
        print("SQLAlchemy 初始化完成。")
    except Exception as init_db_err:
        print(f"错误：SQLAlchemy 初始化失败：{init_db_err}")
        DB_INIT_SUCCESS = False

# --- 全局 RAG 模型实例 (所有用户共享) ---


@app.errorhandler(500)
def handle_internal_server_error(e):
    """
    当服务器发生任何未捕获的内部错误时，确保返回JSON格式的响应。
    """
    # 在后端日志中打印详细的错误堆栈信息，方便调试
    traceback.print_exc()
    return jsonify(error="服务器内部发生错误，请检查后端日志获取详细信息。"), 500

# =========================================
# 装饰器
# =========================================
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
                return jsonify({"error": "需要认证", "login_required": True}), 401
            return redirect(url_for('login_page', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- 原始文件路径和函数 ---

VECTOR_FILE_PATH = '/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/vectorrag/analysis_retrieval___top5_2024-11-26_21-32-23.json'
GRAPH_FILE_PATH = '/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_retrieval_merged.json'
EVIDENCE_FILE_PATH = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb_evidence.json"

#####按行逐个读取
def load_all_items(file_path):
    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    items.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return items

##加载响应的id数据
def load_and_filter_data(file_path, item_id):
    try:
        item_id_int = int(item_id)
    except (ValueError, TypeError):
        return None
    data = load_all_items(file_path)
    return next((item for item in data if item.get('id') == item_id_int), None)

def find_right_arrow(s):
    """
    查找字符串中所有 "->" 的起始位置
    """
    right_arrow_positions = []
    i = 0
    while i < len(s) - 1:  # 确保不会越界
        if s[i:i+2] == "->":
            right_arrow_positions.append(i)
            i += 2  # 跳过这两个字符，避免重复查找
        else:
            i += 1
    return right_arrow_positions

def find_left_arrow(s):
    """
    查找字符串中所有 "<-" 的起始位置
    """
    left_arrow_positions = []
    i = 0
    while i < len(s) - 1:  # 确保不会越界
        if s[i:i+2] == "<-":
            left_arrow_positions.append(i)
            i += 2  # 跳过这两个字符，避免重复查找
        else:
            i += 1
    return left_arrow_positions

#获取所有的-的位置，但它不一定是关系的分隔符
def get_all_dash(s):
    dash_positions = []
    i = 0
    while i < len(s):
        if s[i] == "-":
            # 检查当前位置是否属于箭头的一部分
            if i > 0 and ((s[i:i+2] == "->") or (s[i-1:i+1] == "<-")):
                i += 1  # 跳过整个箭头（两个字符），避免误判 "-" 为单独的 "-"
                continue
            dash_positions.append(i)
        i += 1
    return dash_positions

def find_dash_positions(s,all_dash):
    dash_positions = []
    for i in all_dash:
        # 边界检查
        is_space_before = i > 0 and s[i-1] == " "
        is_space_after = i < len(s) - 1 and s[i+1] == " "
        if is_space_before or is_space_after:
            dash_positions.append(i)
    return dash_positions

def split_relation(rel_seq):
    # This is a complex function. A simplified placeholder is used here.
    # Replace with your actual implementation.
    parts = []
    match = re.match(r'(.+?)\s*-\s*(.+?)\s*->\s*(.+)', rel_seq)
    if match:
        parts.append(match.groups())
    return parts

def convert_rel_to_triplets(retrieve_results):
    triples = set()
    if retrieve_results:
        for rel_seq in retrieve_results:
            parsed_triples = split_relation(rel_seq)
            for t in parsed_triples:
                if len(t) == 3:
                    triples.add(t)
    return list(triples)

def triples_to_json(triples, evdience_entity, evdience_path):
    json_result = {'edges': [], 'nodes': [], 'highlighted-edge': [], 'highlighted-node': []}
    node_set = set()
    colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF"]
    
    for i, triple in enumerate(triples):
        source, relationship, destination = triple
        edge_id = f"e{i}"
        color = colors[i % len(colors)]
        
        edge_data = {'id': edge_id, 'label': relationship, 'source': source, 'target': destination, 'color': color}
        json_result['edges'].append({'data': edge_data})

        if source not in node_set:
            node_set.add(source)
            json_result['nodes'].append({'data': {'id': source, 'label': source, 'color': color}})
        if destination not in node_set:
            node_set.add(destination)
            json_result['nodes'].append({'data': {'id': destination, 'label': destination, 'color': color}})
            
    return json_result

def get_evidence(file_path, item_id):
    # Placeholder implementation
    return [], []

# =========================================
# Flask 路由
# =========================================

# --- 基本页面路由 ---
@app.route('/')
@login_required
def index(): 
    return render_template('demo.html')

@app.route('/login', methods=['GET'])
def login_page():
    if 'user_id' in session: return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    if 'user_id' in session: return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/analysis')
@login_required
def analysis(): return render_template('analysis.html')

# --- 认证 API ---
@app.route('/api/register', methods=['POST'])
def api_register():
    if not request.is_json: return jsonify({"error": "请求必须是 JSON"}), 415
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone') # 假设前端会传来phone

    if not all([username, email, password]):
        return jsonify({"error": "用户名、邮箱和密码是必需的！"}), 400

    try:
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "用户名已存在"}), 409
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "邮箱已被注册"}), 409

        new_user = User(username=username, email=email, phone=phone)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "注册成功！请登录。"}), 201
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "注册过程中发生内部错误。"}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    if not request.is_json: return jsonify({"error": "请求必须是 JSON"}), 415
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "用户名和密码是必需的！"}), 400
    
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        session.clear()
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({"message": "登录成功！"}), 200
    else:
        return jsonify({"error": "无效的用户名或密码"}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    session.clear()
    return jsonify({"message": "注销成功"}), 200

@app.route('/api/check-auth', methods=['GET'])
def api_check_auth():
    """检查用户登录状态"""
    if 'user_id' in session: return jsonify({"logged_in": True, "user_id": session['user_id'], "username": session.get('username')}), 200
    else: return jsonify({"logged_in": False}), 200


# --- 格式化 SSE 输出 ---
def sse_pack(data_dict: dict) -> str:
    """将字典格式化为 Server-Sent Event (SSE) data 字段"""
    return f"data: {json.dumps(data_dict)}\n\n"

@app.route('/read-file', methods=['GET'])
def read_file():
    try:
        # 设定文件路径 (这个路径可能是固定的，或者应该作为参数?)
        file_path = '/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/test.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
        result = []
        for item in data:
            result.append({
                'id': item.get('id'),
                'question': item.get('question'),
                'answer': item.get('answer'),
                'hybrid_response': item.get('hybrid_response'),
                'type' : item.get('type')
            })
        return jsonify({'content': result})
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return jsonify({'error': str(e)}), 500
    
#通过用户名字在数据表中指定，显示出该用户的所有对话历史table    
# 返回示例
# {
#   "user_name": "Alice",
#   "user_id": "u_123",
#   "history_tables": ["test1", "eval_0429", "final"]
# }


# --- 历史记录表管理 API ---
@app.route('/create-history-session', methods=['POST'])
@login_required
def create_history_session():
    """为当前用户创建一个新的历史记录表"""
    user_id = session['user_id']
    data = request.get_json()
    session_name_suffix = data.get('sessionName')

    if not session_name_suffix:
        return jsonify({"success": False, "message": "会话名称 (sessionName) 不能为空"}), 400

    if not mysql:
        return jsonify({"success": False, "message": "数据库连接不可用"}), 503

    try:
        print(f"用户 {user_id} 正在尝试创建历史表，后缀为: {session_name_suffix}")
        mysql.add_history_table(str(user_id), session_name_suffix)
        return jsonify({"success": True, "message": f"历史表 '{session_name_suffix}' 创建成功"}), 201
    except ValueError as ve:
        return jsonify({"success": False, "message": str(ve)}), 400
    except Exception as e:
        if "already exists" in str(e).lower():
            return jsonify({"success": False, "message": "该名称的历史表已存在"}), 409
        traceback.print_exc()
        return jsonify({"success": False, "message": "创建历史表时发生服务器内部错误"}), 500

@app.route('/delete-history-session', methods=['DELETE'])
@login_required
def delete_history_session():
    """删除当前用户的指定历史记录表"""
    user_id = session['user_id']
    data = request.get_json()
    session_name_suffix = data.get('sessionName')

    if not session_name_suffix:
        return jsonify({"success": False, "message": "会话名称 (sessionName) 不能为空"}), 400

    if not mysql:
        return jsonify({"success": False, "message": "数据库连接不可用"}), 503

    try:
        print(f"用户 {user_id} 正在尝试删除历史表，后缀为: {session_name_suffix}")
        deleted = mysql.del_history_table(str(user_id), session_name_suffix)
        if deleted:
            return jsonify({"success": True, "message": f"历史表 '{session_name_suffix}' 已删除"}), 200
        else:
            return jsonify({"success": False, "message": "历史表不存在，无法删除"}), 404
    except ValueError as ve:
        return jsonify({"success": False, "message": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": "删除历史表时发生服务器内部错误"}), 500


# --- 历史记录数据获取 API ---
@app.route("/get-history-tables", methods=["GET"])
@login_required
def get_history_tables():
    user_id = session.get("user_id")
    print(f"--- [日志] 接收到用户 {user_id} 的 /get-history-tables 请求 ---")
    if not mysql:
        return jsonify({"error": "数据库连接不可用"}), 503
    try:
        suffixes = mysql.get_user_history_suffixes(str(user_id))
        print(f"--- [日志] 查询成功，为用户 {user_id} 找到 {len(suffixes)} 个历史表 ---")
        return jsonify({"user_id": user_id, "history_tables": suffixes})
    except Exception as e:
        print(f"--- [错误] 在 /get-history-tables 中查询数据库时出错: {e} ---")
        traceback.print_exc()
        # 错误将由全局错误处理器捕获并返回JSON
        raise e

@app.route("/get-history-entries")
@login_required
def get_history_entries():
    table_suffix = request.args.get("table_suffix")
    user_id = session.get("user_id")

    print("##################################",user_id)

    if not table_suffix or not user_id:
        return jsonify({"entries": [], "error": "缺少参数 table_suffix 或用户未登录"}), 400

    table_name = f"user{user_id}_history_{table_suffix}"

    try:
        cursor = mysql.cursor
        cursor.execute(f"SELECT id, query, answer, type, vector_response, graph_response, hybrid_response, no_rag_response FROM `{table_name}` ORDER BY created_at DESC LIMIT 100")
        rows = cursor.fetchall()
        entries = [{"id": r[0], "query": r[1], "answer": r[2], "type": r[3], "vector_response": r[4], "graph_response": r[5], "hybrid_response": r[6], "no_rag_response": r[7]} for r in rows]
        return jsonify({"entries": entries})
    except Exception as e:
        if "doesn't exist" in str(e):
             return jsonify({"entries": [], "error": "指定的历史表不存在"}), 404
        traceback.print_exc()
        return jsonify({"entries": [], "error": str(e)}), 500


# --- RAG 核心交互与数据检索 API ---
# @app.route('/load_model', methods=['POST'])
# @login_required
# def load_model():
#     global current_model
#     try:
#         data = request.json
#         mode = data.get("mode")
#         if mode == "Stop":
#             if current_model is not None:
#                 print("Stopping the current model.")
#                 current_model.close()
#                 current_model = None
#                 return jsonify({"status": "success", "message": "模型已停止"}), 200
#             else:
#                 return jsonify({"status": "error", "message": "没有正在运行的模型"}), 400
        
#         model_name = data.get("model_name")
#         key = data.get("key")
#         dataset_info = data.get("dataset")
#         hop = dataset_info.get('hop')
#         type = dataset_info.get('type')
#         entity = dataset_info.get('entity')
#         dataset = dataset_info.get('dataset')
#         session_suffix = dataset_info.get('session')
#         dataset_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"
#         if not key: key = "ollama"
        
#         print(f"正在加载模型: {model_name}, 数据集: {dataset}")
#         current_model = Demo_chat(model_name=model_name, api_key=key, dataset_name=dataset, dataset_path=dataset_path, path_name=session_suffix)

#         def generate():
#             yield json.dumps({"status": "start", "message": f"模型 {model_name} (数据集: {dataset}) 加载成功"}) + "\n"
#             for item_data in current_model.new_history_chat(mode=mode):
#                 yield json.dumps({"status": "processing", "item_data": item_data}) + "\n"
#             yield json.dumps({"status": "complete", "message": "所有项目处理完成"}) + "\n"

#         return Response(generate(), mimetype='text/event-stream')
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "message": str(e)}), 500
    

@app.route('/load_model', methods=['POST'])
@login_required
def load_model_multi():
    current_model: Union[Demo_chat, None] = None
    user_id = session.get("user_id")
    try:
        mysql = MySQLManager()
        data = request.json
        mode = data.get("mode")
        if mode == "Stop":
            if current_model is not None:
                print("Stopping the current model.")
                current_model.close()
                current_model = None
                return jsonify({"status": "success", "message": "模型已停止"}), 200
            else:
                return jsonify({"status": "error", "message": "没有正在运行的模型"}), 400

        model_name = data.get("model_name")
        key = data.get("key")
        dataset_info = data.get("dataset")
        hop = dataset_info.get('hop')
        type = dataset_info.get('type')
        entity = dataset_info.get('entity')
        dataset = dataset_info.get('dataset')
        table_suffix = dataset_info.get('session')
        dataset_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"
        if not key: key = "ollama"
        print(f"正在加载模型: {model_name}, 数据集: {dataset}")
        current_model = Demo_chat(model_name=model_name, api_key=key, dataset_name=dataset, dataset_path=dataset_path, path_name=table_suffix)

        def generate():
            yield json.dumps({"status": "start", "message": f"模型 {model_name} (数据集: {dataset}) 加载成功"}) + "\n"
            for item_data in current_model.new_history_chat(mode=mode):
                print(item_data)
                mysql.insert_record_to_history_table(user_id=user_id,table_suffix=table_suffix,record=item_data)
                yield json.dumps({"status": "processing", "item_data": item_data}) + "\n"
            yield json.dumps({"status": "complete", "message": "所有项目处理完成"}) + "\n"

       

        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    
# 用于用户自己输入问题进行问答
@app.route('/ask', methods=['POST'])
@login_required
def user_ask_question():
    user_id = session.get("user_id")
    data = request.json
    query = data.get("query", "").strip()
    model_name = data.get("model_name")
    key = data.get("key")
    dataset_info = data.get("dataset")
    hop = dataset_info.get('hop')
    type = dataset_info.get('type')
    entity = dataset_info.get('entity')
    dataset = dataset_info.get('dataset')
    table_suffix = dataset_info.get('session')
    dataset_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"

    if not query:
        return jsonify({"status": "error", "message": "问题不能为空"}), 400

    try:
        current_model = Demo_chat(model_name=model_name, api_key=key, dataset_name=dataset, dataset_path=dataset_path, path_name=table_suffix)

        item_data = current_model.user_query(query=query,user_id=user_id)

        mysql = MySQLManager()
        mysql.insert_record_to_history_table(
            user_id=user_id,
            table_suffix=table_suffix,
            record=item_data
        )

        return jsonify({"status": "success", "item_data": item_data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/get-vector/<item_id>', methods=['GET'])
@login_required
def get_vector(item_id):
    mysql = MySQLManager()
    user_id = session.get("user_id")
    table_suffix = request.args.get("sessionName")

    if not table_suffix:
        return jsonify({'error': '缺少参数 tableSuffix'}), 400

    table_name = f"user{user_id}_history_{table_suffix}"
    try:
        query_sql = f"SELECT vector_retrieval_result FROM `{table_name}` WHERE id = %s"
        cursor = mysql.cursor
        cursor.execute(query_sql, (item_id,))
        result = cursor.fetchone()

        if not result:
            return jsonify({'error': f'未找到 ID 为 {item_id} 的记录'}), 404

        raw_retrieval = result[0]
        try:
            retrieval_chunks = json.loads(raw_retrieval) if raw_retrieval else []
        except json.JSONDecodeError:
            retrieval_chunks = [raw_retrieval]

        return jsonify({'id': item_id, 'chunks': retrieval_chunks})
    except Exception as e:
        return jsonify({'error': f'查询失败: {str(e)}'}), 500

@app.route('/get-graph/<item_id>', methods=['GET'])
@login_required
def get_graph(item_id):
    mysql = MySQLManager()
    user_id = session.get("user_id")
    table_suffix = request.args.get("sessionName")

    if not user_id or not table_suffix:
        return jsonify({'error': '缺少参数 tableSuffix 或用户未登录'}), 400

    table_name = f"user{user_id}_history_{table_suffix}"
    try:
        cursor = mysql.cursor
        query_sql = f"SELECT graph_retrieval_result FROM `{table_name}` WHERE id = %s"
        cursor.execute(query_sql, (item_id,))
        result = cursor.fetchone()

        if not result:
            return jsonify({'error': f'未找到 ID 为 {item_id} 的记录'}), 404

        raw_graph_data = result[0]
        try:
            graph_data = json.loads(raw_graph_data) if isinstance(raw_graph_data, str) else raw_graph_data
        except json.JSONDecodeError:
            return jsonify({'error': '图数据格式解析失败'}), 500

        evidence_entity, evidence_path = get_evidence(EVIDENCE_FILE_PATH, item_id)
        triples = convert_rel_to_triplets(graph_data)

        if not triples:
            return jsonify({'edges': [], 'nodes': [], 'highlighted-edge': [], 'highlighted-node': []})

        json_result = triples_to_json(triples, evidence_entity, evidence_path)
        return jsonify(json_result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'数据库查询错误: {str(e)}'}), 500





# --- 检索详情 API 本地版本，从文件路径中读取 ---
# @app.route('/get-vector/<item_id>', methods=['GET'])
# @login_required
# def get_vector(item_id):
#     # 获取与 item_id 相关的 vector 数据
#     session_name = request.args.get("sessionName")
#     dataset_name = request.args.get("datasetName")


#     session_file = os.path.abspath(
#         os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name,f"{session_name}.json")
#     )
#     filtered_data = load_and_filter_data(session_file, item_id)
#     retrieval_result = []
#     if filtered_data:
#          # 确保前端期望的 'chunks' 键存在
#         if 'vector_retrieval_result' in filtered_data and isinstance(filtered_data['vector_retrieval_result'], list):
            
#              # 简单地将 retrieve_results 的值（假设是文本列表）转换为 chunk 对象
#              for text_list in filtered_data['vector_retrieval_result']:
#                  retrieval_result.append(text_list)
        
#         # print("#######retrieval_result########",retrieval_result)
#         result = {
#             'id': item_id,
#             'chunks': retrieval_result
            
#         }

#         return jsonify(result)  # 返回处理后的数据
#     else:
#         return jsonify({'error': f'Item not found or invalid ID format for vector lookup: {item_id}'}), 404

# @app.route('/get-graph/<item_id>', methods=['GET'])
# @login_required
# def get_graph(item_id):
#     session_name = request.args.get("sessionName")
#     dataset_name = request.args.get("datasetName")


#     session_file = os.path.abspath(
#         os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name,f"{session_name}.json")
#     )
#     filtered_data = load_and_filter_data(session_file, item_id)
#     if filtered_data and 'graph_retrieval_result' in filtered_data:
#         evidence_entity,evidence_path = get_evidence(EVIDENCE_FILE_PATH,item_id)
#         # 转换 retrieve_results 为三元组
#         triples = convert_rel_to_triplets(filtered_data["graph_retrieval_result"])
#         print("############triples###########",triples)
#         if not triples:
#              # print(f"Warning: No triples generated for item {item_id}. Returning empty graph.")
#              return jsonify({'edges': [], 'nodes': [], 'highlighted-edge': [], 'highlighted-node': []})

#         json_result = triples_to_json(triples,evidence_entity,evidence_path)
#         # print("================= GRAPH RESPONSE =================") # 原始调试信息
#         # print(json.dumps(json_result, indent=2))                  # 原始调试信息
#         # print("================================================") # 原始调试信息
#         return jsonify(json_result)  # 返回找到的数据
#     elif filtered_data is None:
#          # 如果 load_and_filter_data 返回 None (ID 不是整数或文件找不到)
#         return jsonify({'error': f'Item not found or invalid ID format for graph lookup: {item_id}'}), 404
#     else:
#         # 如果找到了数据但格式不对
#         print(f"Warning: Graph data format error or missing 'retrieve_results' for item {item_id}")
#         return jsonify({'error': f'Data format error for graph item {item_id}'}), 500

# --- 分析 / 建议路由 ---
@app.route('/get_suggestions', methods=['GET'])
def adviser():
    # 假设下面这些文件路径是正确的
    rgb_graph_generation = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_generation___merged.json"
    rgb_graph_retrieval = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_retrieval_merged.json"
    rgb_vector_generation = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/vectorrag/analysis_generation___top5_2024-11-26_21-32-23.json"
    rgb_vector_retrieval = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/vectorrag/analysis_retrieval___top5_2024-11-26_21-32-23.json"

    try: # 添加 try-except 块
        # 假设 statistic_error_cause 函数已经定义且可用
        v_retrieve_error, v_lose_error, v_lose_correct = simulate.statistic_error_cause(rgb_vector_generation, rgb_vector_retrieval, "vector")
        g_retrieve_error, g_lose_error, g_lose_correct = simulate.statistic_error_cause(rgb_graph_generation, rgb_graph_retrieval, "graph")
        suggestions = {
            "vector_retrieve_error": v_retrieve_error, "vector_lose_error": v_lose_error, "vector_lose_correct": v_lose_correct,
            "graph_retrieve_error": g_retrieve_error, "graph_lose_error": g_lose_error, "graph_lose_correct": g_lose_correct,
            "advice": "这里是对用户的建议" # 原始建议文本
        }
        # print(suggestions) # 原始调试信息
        return jsonify(suggestions)
    except AttributeError:
         error_msg = "Error: 'simulate' module or 'statistic_error_cause' function not found or loaded correctly."
         print(error_msg)
         return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"An error occurred while generating suggestions: {e}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route('/get_accuracy', methods=['GET'])
# @login_required
def get_accuracy():
    # 模拟的准确度数据，实际可以从模型或数据库中获取 (原始注释)
    # 注意: 这个接口可能与 /api/analysis_data 重复，考虑是否保留
    try: # 添加 try-except
        # 假设 simulate 模块和相应函数可用
        graph_gen_faithfulness, graph_gen_accuracy = simulate.statistic_graph_generation(simulate.rgb_graph_generation)
        vector_gen_precision, vector_gen_faithfulness, vector_gen_accuracy = simulate.statistic_vector_generation(simulate.rgb_vector_generation)
        data = {
            "vector_accuracy": round(vector_gen_accuracy * 100, 1),
            "graph_accuracy": round(graph_gen_accuracy * 100, 1),
            "hybrid_accuracy": 85 # 原始占位符
        }
        return jsonify(data)
    except AttributeError:
        error_msg = "Error: 'simulate' module or statistics functions not found or loaded correctly."
        print(error_msg)
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"An error occurred while getting accuracy: {e}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route('/api/analysis_data', methods=['GET'])
# @login_required
def get_analysis_data():
    try: # 添加 try-except
        # 假设 simulate 模块和相应文件/函数可用
        graph_gen_faithfulness, graph_gen_accuracy = simulate.statistic_graph_generation(simulate.rgb_graph_generation)
        graph_ret_recall, graph_ret_relevance = simulate.statistic_graph_retrieval(simulate.rgb_graph_retrieval)
        vector_gen_precision, vector_gen_faithfulness, vector_gen_accuracy = simulate.statistic_vector_generation(simulate.rgb_vector_generation)
        vector_ret_precision, vector_ret_relevance, vector_ret_recall = simulate.statistic_vector_retrieval(simulate.rgb_vector_retrieval)
        hybrid_precision, hybrid_faithfulness, hybrid_accuracy, hybrid_relevance, hybrid_recall = simulate.statistic_hybrid_generation(simulate.rgb_vector_retrieval)

        # 临时统计
        error_stats_vectorrag = {'None Result': 37.7, 'Lack Information': 14.2, 'Noisy': 7.1, 'Other': 41.0}
        error_stats_graphrag = {'None Result': 69.4, 'Lack Information': 8.3, 'Noisy': 5.6, 'Other': 16.7}
        error_stats_hybridrag = {'None Result': 36.8, 'Lack Information': 9.1, 'Noisy': 9.1, 'Other': 45.0}

        # 临时 hybridrag 统计
        eval_metrics_vectorrag = {'precision': vector_ret_precision, 'relevance': vector_ret_relevance, 'recall': vector_ret_recall, 'faithfulness': vector_gen_faithfulness, 'accuracy': vector_gen_accuracy}
        eval_metrics_graphrag = {'precision': graph_ret_relevance, 'relevance': graph_ret_relevance, 'recall': graph_ret_recall, 'faithfulness': graph_gen_faithfulness, 'accuracy': graph_gen_accuracy}
        eval_metrics_hybridrag = {'precision': hybrid_precision, 'relevance': hybrid_relevance, 'recall': hybrid_recall, 'faithfulness': hybrid_faithfulness, 'accuracy': hybrid_accuracy}
        
        # 临时 hybridrag 统计
        analysis_data = {
            "accuracy": { "graphrag": round(graph_gen_accuracy * 100, 1), "vectorrag": round(vector_gen_accuracy * 100, 1), "hybridrag": round(hybrid_accuracy * 100, 1) },
            "errorStatistics": { "vectorrag": error_stats_vectorrag, "graphrag": error_stats_graphrag, "hybridrag": error_stats_hybridrag },
            "evaluationMetrics": { "vectorrag": eval_metrics_vectorrag, "graphrag": eval_metrics_graphrag, "hybridrag": eval_metrics_hybridrag }
        }
        # print(analysis_data) # 原始调试信息
        return jsonify(analysis_data)
    except AttributeError:
         error_msg = "Error: 'simulate' module or statistics functions/files not found or loaded correctly."
         print(error_msg)
         return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"An error occurred while generating analysis data: {e}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

# --- 创建数据库表的命令 ---
@app.cli.command("create-db")
def create_db_command():
    """根据 models.py 创建 SQLAlchemy 管理的表 (例如 users)"""
    if DB_INIT_SUCCESS:
        with app.app_context():
            try: 
                db.create_all()
                print("SQLAlchemy 管理的表（如 user）创建成功。")
            except Exception as e: 
                print(f"错误：创建数据库表时出错：{e}")
    else: 
        print("错误：无法创建表，数据库初始化失败。")

# --- 主程序入口 ---
if __name__ == '__main__':
    ui_port = int(os.environ.get('FLASK_PORT', 5000))
    print(f" * 启动 Flask 应用于 http://0.0.0.0:{ui_port}")
    if not RAG_CORE_LOADED: print(" * 警告：RAG 核心未加载。")
    if not DB_INIT_SUCCESS: print(" * 错误：数据库未正确配置或初始化失败。")
    if not mysql: print(" * 错误: MySQLManager 未能成功连接，历史记录功能将不可用。")
    # 在开发时建议开启 debug=True
    app.run(host='0.0.0.0', port=ui_port, debug=False, threaded=True)
