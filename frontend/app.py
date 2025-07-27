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
#from user import User



current_dir = os.getcwd()

# 获取当前工作目录的上级目录
parent_dir = os.path.dirname(current_dir)

# 拼接出 'backend' 文件夹的路径
backend_dir = os.path.join(parent_dir, 'backend')

# 将 'backend' 目录添加到 sys.path 中
sys.path.append(backend_dir)
from  llmragenv.llmrag_env import LLMRAGEnv
from llmragenv.demo_chat import *
from evaluator import simulate
from llmragenv.demo_chat import Demo_chat
import threading
import traceback

# --- RAG 核心逻辑导入 ---
RAG_CORE_LOADED = False # <--- 检查！！是否在 try 之前有这行初始化？
try:
    from llmragenv.demo_chat import Demo_chat
    RAG_CORE_LOADED = True # 成功导入则设为 True
except ImportError as e:
    print(f"警告：无法导入 RAG 核心逻辑 'llmragenv.demo_chat'：{e}")
    # 此处 RAG_CORE_LOADED 保持为 False
    Demo_chat = None
from evaluator import simulate

# =========================================
# 初始化与配置
# =========================================
load_dotenv() # 加载 .env 文件

app = Flask(__name__, instance_relative_config=True) # 初始化 Flask 应用

# --- 应用配置 ---
# 必须设置强密钥用于 Session 安全
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-insecure-key-change-me')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 数据库 URI 配置 (依赖环境变量)
db_uri = os.environ.get('DATABASE_URL', None)
if db_uri is None:
    print("错误：启动必需的 DATABASE_URL 环境变量未设置！请设置指向 MySQL 的连接字符串。")
    app.config['SQLALCHEMY_DATABASE_URI'] = None
else:
     app.config['SQLALCHEMY_DATABASE_URI'] = db_uri

# --- 初始化数据库 ---
# 直接初始化，如果 URI 无效或 db 未定义，会在执行时出错
try:
    db.init_app(app)
    print("数据库初始化完成。")
    DB_INIT_SUCCESS = True
except Exception as init_db_err:
     print(f"错误：初始化数据库失败：{init_db_err}")
     DB_INIT_SUCCESS = False # 标记数据库初始化失败

# --- 全局 RAG 模型实例 (所有用户共享) ---
current_model: Union[Demo_chat, None] = None # 类型提示
current_model_dataset_info = {}

# =========================================
# 装饰器
# =========================================
def login_required(f):
    """检查用户是否登录，未登录则重定向或返回401。"""
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

##加载响应的id数据
def load_and_filter_data(file_path, item_id):
    # 注意: 此函数期望 item_id 是整数，用于查找静态文件。
    # 如果前端在 API 模式下传递 UUID，这里会查找失败。
    try:
        item_id_int = int(item_id)
    except (ValueError, TypeError):
         # print(f"Warning: load_and_filter_data: Could not convert item_id '{item_id}' to integer.") # 可选的调试信息
         return None # 无法用非整数ID在此函数中查找

    try:
            data = load_all_items(file_path)
            # 通过 item_id 查找对应的元素
            filtered_data = next((item for item in data if item.get('id') == item_id_int), None)
            return filtered_data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}.")
        return None

#####按行逐个读取
def load_all_items(file_path):
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return items

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
    # --- 保持原始的 split_relation 逻辑 ---
    # 注意: 原始逻辑比较复杂，可能需要根据实际关系字符串格式进行调试或简化
    parts = []
    rel_seq = rel_seq.strip()
    all_dash = get_all_dash(rel_seq)
    right_arrows = find_right_arrow(rel_seq)
    left_arrows = find_left_arrow(rel_seq)
    dash_positions = find_dash_positions(rel_seq,all_dash)
    arrows_index = sorted(list(set(right_arrows+left_arrows))) # 修正: 使用 set 去重

    if len(arrows_index) == 1:
        # 处理单箭头情况
        arrow_pos = arrows_index[0]
        # 确保 dash_positions 非空
        dash_pos = dash_positions[0] if dash_positions else -1
        if dash_pos == -1: return parts # 无法分割

        if arrow_pos in right_arrows:
            source = rel_seq[:dash_pos].strip() # 使用第一个 dash
            rel = rel_seq[dash_pos+1:arrow_pos].strip()
            dst = rel_seq[arrow_pos+2:].strip()
            if source and rel and dst: parts.append((source,rel,dst))
        else: # left_arrow
            dst = rel_seq[:arrow_pos].strip()
            # 假设关系在箭头和 dash 之间
            rel = rel_seq[arrow_pos+2:dash_pos].strip() # 使用第一个 dash
            source = rel_seq[dash_pos+1:].strip()
            if source and rel and dst: parts.append((source,rel,dst))
        return parts

    # 原始多跳分解逻辑 (非常复杂，此处保留结构，但可能需要大量调试)
    # 注意: 索引处理和边界条件需要非常小心
    elif len(arrows_index) > 1 and len(dash_positions) >= len(arrows_index):
        i = 0
        try: # 增加 try-except 块捕获索引错误
            for arrow_pos in arrows_index:
                if arrow_pos in right_arrows:
                    if i == 0:
                        source = rel_seq[:dash_positions[0]].strip()
                        rel = rel_seq[dash_positions[0]+1:arrow_pos].strip()
                        dst = rel_seq[arrow_pos+2:min(dash_positions[1],arrows_index[1])].strip()
                    elif i == len(arrows_index)-1:
                        source = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):dash_positions[-1]].strip()
                        rel = rel_seq[dash_positions[-1]+1:arrow_pos].strip()
                        dst = rel_seq[arrow_pos+2:].strip()
                    else:
                        source = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):dash_positions[i]].strip()
                        rel = rel_seq[dash_positions[i]+1:arrow_pos].strip()
                        dst = rel_seq[arrow_pos+2:min(dash_positions[i+1],arrows_index[i+1])].strip()
                elif arrow_pos in left_arrows: # left_arrow
                     if i == 0:
                        dst = rel_seq[:arrow_pos].strip()
                        rel = rel_seq[arrow_pos+2:dash_positions[i]].strip()
                        source = rel_seq[dash_positions[i]+1:min(dash_positions[i+1],arrows_index[i+1])].strip()
                     elif i == len(arrows_index)-1:
                        dst = rel_seq[max(arrows_index[i-1]+2,dash_positions[i-1]+1):arrow_pos].strip()
                        rel = rel_seq[arrow_pos+2:dash_positions[i]].strip() # 使用第 i 个 dash
                        source = rel_seq[dash_positions[i]+1:].strip()
                     else:
                        dst = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):arrow_pos].strip()
                        rel = rel_seq[arrow_pos+2:dash_positions[i]].strip()
                        source = rel_seq[dash_positions[i]+1:min(dash_positions[i+1],arrows_index[i+1])].strip() # 使用第 i+1 个元素
                else: continue # Should not happen

                if source and rel and dst: parts.append((source, rel, dst))
                i += 1
        except IndexError as e:
            print(f"Error parsing multi-hop relation '{rel_seq}' due to index error: {e}")
            # 可能返回部分解析结果或空列表
            return parts # 返回已解析的部分

    elif not parts and len(rel_seq) > 0: # 如果无法解析
        print(f"Warning: Could not parse relation (complex or unexpected format): {rel_seq}")

    return parts

def convert_rel_to_triplets(retrieve_results):
    triples = set()
    for rel_seq in retrieve_results:
        parsed_triples = split_relation(rel_seq)
        for t in parsed_triples:
                if len(t) == 3: # 确保是有效三元组
                    triples.add(t)
    return list(triples)

def triples_to_json(triples,evdience_entity,evdience_path):
    # --- 保持原始的 triples_to_json 逻辑 ---
    colors = [ "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#B5EAD7", "#ECC5FB", "#FFC3A0", "#FF9AA2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2", "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2", "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2", "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1" ]
    json_result = {'edges': [], 'nodes': [],'highlighted-edge':[],'highlighted-node':[]}
    node_set = set()
    import random
    # print(f"Triples:{triples}")

    for i, triple in enumerate(triples): # 使用 enumerate 获取索引
        if len(triple) != 3: continue # 跳过无效三元组
        source, relationship, destination = triple
        edge_id = f"e{i}" # 为边分配唯一 ID
        color = colors[random.randint(0, len(colors)-1)] # 修正随机颜色索引

        # 添加边
        edge_data = { 'id': edge_id, 'label': relationship, 'source': source, 'target': destination, 'color': color } # 添加 id
        json_result['edges'].append({ 'data': edge_data })
        if relationship in evdience_path:
            json_result['highlighted-edge'].append({'data': edge_data})

        # 添加节点 (避免重复)
        if source not in node_set:
            node_data_source = {'id': source, 'label': source, 'color': color} # 初始颜色
            json_result['nodes'].append({'data': node_data_source})
            if source in evdience_entity:
                json_result['highlighted-node'].append({'data': node_data_source})
            node_set.add(source)
        elif source in evdience_entity and not any(n['data']['id'] == source for n in json_result['highlighted-node']):
            # 如果节点已存在但未高亮，则高亮它
             existing_node = next((n for n in json_result['nodes'] if n['data']['id'] == source), None)
             if existing_node: json_result['highlighted-node'].append(existing_node)

        if destination not in node_set:
            node_data_dest = {'id': destination, 'label': destination, 'color': color}
            json_result['nodes'].append({'data': node_data_dest})
            if destination in evdience_entity:
                json_result['highlighted-node'].append({'data': node_data_dest})
            node_set.add(destination)
        elif destination in evdience_entity and not any(n['data']['id'] == destination for n in json_result['highlighted-node']):
             existing_node = next((n for n in json_result['nodes'] if n['data']['id'] == destination), None)
             if existing_node: json_result['highlighted-node'].append(existing_node)

    return json_result

def get_evidence(file_path,item_id):
    # 注意: 同样存在 item_id 类型问题
    try:
        item_id_int = int(item_id)
    except (ValueError, TypeError):
        # print(f"Warning: get_evidence: Could not convert item_id '{item_id}' to integer.")
        return [], []

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        e = next((item for item in data if item.get('id') == item_id_int), None)

        entity = set() # 使用集合去重
        path = set()
        if e and "merged_triplets" in e and isinstance(e["merged_triplets"], list):
            # 遍历 evidence 中的每个三元组列表
            for t_list in e["merged_triplets"]:
                if isinstance(t_list, list):
                    # 遍历列表中的每个三元组
                    for triple in t_list:
                        if isinstance(triple, list) and len(triple) == 3:
                            entity.add(triple[0])
                            path.add(triple[1])
                            entity.add(triple[2])
            return list(entity), list(path) # 转换回列表返回
        else:
            # print(f"No valid 'merged_triplets' found for id {item_id_int} in {file_path}")
            return [], []
    except FileNotFoundError:
         print(f"Evidence file not found: {file_path}")
         return [], []
    except Exception as ex:
        print(f"Error processing evidence file {file_path} for id {item_id}: {ex}")
        return [], []

# =========================================
# Flask 路由
# =========================================

# --- 基本页面路由 ---
@app.route('/')
@login_required
def index(): 
    username = session.get('username')
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
    """处理注册请求"""
    if not request.is_json: return jsonify({"error": "请求必须是 JSON"}), 415
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    data = request.get_json(); username, email, phone, password, confirm_password = map(data.get, ['username', 'email', 'phone', 'password', 'confirm_password']); errors = []
    if not all([username, email, phone, password, confirm_password]): errors.append("所有字段都是必需的！")
    if password != confirm_password: errors.append("密码和确认密码不匹配！")
    if not errors:
        try: # 检查用户是否存在
            if User.query.filter_by(username=username).first(): errors.append("用户名已存在")
            if User.query.filter_by(email=email).first(): errors.append("邮箱已被注册")
            if phone and User.query.filter_by(phone=phone).first(): errors.append("手机号已被注册")
        except Exception as e: return jsonify({"error": "检查用户信息时数据库出错。"}), 500
    if errors: return jsonify({"error": ", ".join(errors)}), 400
    try: # 创建用户
        new_user = User(username=username, email=email, phone=phone); new_user.set_password(password)
        db.session.add(new_user); db.session.commit()
        return jsonify({"message": "注册成功！请登录。"}), 201
    except Exception as e: db.session.rollback(); return jsonify({"error": "注册过程中发生内部错误。"}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """处理登录请求"""
    if not request.is_json: return jsonify({"error": "请求必须是 JSON"}), 415
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    data = request.get_json(); username, password = data.get('username'), data.get('password')
    if not username or not password: return jsonify({"error": "用户名和密码是必需的！"}), 400
    try:
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session.clear(); session['user_id'] = user.id; session['username'] = user.username
            
            return jsonify({"message": "登录成功！"}), 200
        else: return jsonify({"error": "无效的用户名或密码"}), 401
    except Exception as e: return jsonify({"error": "登录过程中发生内部错误。"}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """处理登出请求"""
    session.clear()
    return jsonify({"message": "注销成功"}), 200

@app.route('/api/check-auth', methods=['GET'])
def api_check_auth():
    """检查用户登录状态"""
    if 'user_id' in session: return jsonify({"logged_in": True, "user_id": session['user_id'], "username": session.get('username')}), 200
    else: return jsonify({"logged_in": False}), 200

# --- 历史记录 / 会话 API ---
@app.route('/api/sessions', methods=['GET'])
@login_required
def list_sessions():
    """获取当前用户会话列表"""
    user_id = session['user_id']
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    try:
        user_sessions = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.create_time.desc()).all()
        return jsonify([{"id": s.id, "name": s.session_name, "create_time": s.create_time.isoformat()+'Z'} for s in user_sessions])
    except Exception as e: return jsonify({"error": "获取会话列表时出错"}), 500

@app.route('/api/sessions', methods=['POST'])
@login_required
def create_session_api():
    """创建新会话"""
    user_id = session['user_id']
    if not request.is_json: return jsonify({"error": "请求必须是 JSON"}), 415
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    data = request.get_json(); session_name = data.get("sessionName", "").strip() or f"会话 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    try: new_session = ChatSession(user_id=user_id, session_name=session_name); db.session.add(new_session); db.session.commit(); return jsonify({"id": new_session.id, "name": new_session.session_name, "create_time": new_session.create_time.isoformat()+'Z'}), 201
    except Exception as e: db.session.rollback(); return jsonify({"error": "创建会话时发生内部错误"}), 500

@app.route('/api/sessions/<int:session_id>/history', methods=['GET'])
@login_required
def get_session_history(session_id):
    """获取指定会话的历史记录 (验证所有权)"""
    user_id = session['user_id']
    if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503
    try:
        chat_session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not chat_session: return jsonify({"error": "会话未找到或无权访问"}), 404
        messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp.asc()).all()
        return jsonify([msg.to_dict() for msg in messages])
    except Exception as e: return jsonify({"error": "获取历史记录时发生内部错误"}), 500
    


# --- RAG 核心交互 API ---
@app.route('/load_model', methods=['POST'])
@login_required
def load_model():
    global current_model
    try:
        data = request.json
        mode = data.get("mode")
        if mode == "Stop":
            # 假设 Demo_chat 中有 close() 方法来停止模型的操作
            if current_model is not None:
                print("Stopping the current model.")
                current_model.close()  # 调用 Demo_chat 中的关闭模型的方法
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
        session = dataset_info.get('session')
        dataset_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"  
        if key == "" or key is None:  # 处理空或 None 的 key
            key = "ollama"  # 默认 key
        print(f"Received /load_model: model={model_name}, key={'<default_ollama>' if key=='ollama' else '<provided>'}, dataset={dataset}")

        if not model_name:
            return jsonify({"status": "error", "message": "缺少模型名称"}), 400
        if not dataset:
            print("Warning: Dataset parameter is missing in /load_model request.")
        
        print(f"正在加载模型: {model_name} (API Key: {'*'*(len(key)-3)+key[-3:] if key != 'ollama' and key else 'ollama'}, 数据集: {dataset})")
        current_model = Demo_chat(model_name=model_name, api_key=key, dataset_name=dataset, dataset_path=dataset_path, path_name=session)

        

        def generate():
            # 发送初始状态
            yield json.dumps({"status": "start", "message": f"模型 {model_name} (数据集: {dataset}) 加载成功"}) + "\n"
            
            # 处理每个项目并立即发送
            for item_data in current_model.new_history_chat(mode = mode):
                yield json.dumps({"status": "processing", "item_data": item_data}) + "\n"
            
            # 发送完成状态
            yield json.dumps({"status": "complete", "message": "所有项目处理完成"}) + "\n"

        return Response(generate(), mimetype='text/event-stream')


    except Exception as e:
        print(f"加载模型时出错: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# @app.route('/generate', methods=['POST'])
# @login_required
# def generate_answers():
#     """处理 RAG 生成请求并保存历史"""
#     global current_model
#     if not RAG_CORE_LOADED or current_model is None: return jsonify({"error": "模型未加载或不可用"}), 501
#     if not request.is_json: return jsonify({"error": "请求必须是 JSON"}), 415
#     if not DB_INIT_SUCCESS: return jsonify({"error": "数据库服务不可用"}), 503

#     try:
#         data = request.json; user_input, session_id, rag_mode = data.get("input"), data.get("session_id"), data.get("rag_mode", "hybrid")
#         if not user_input or not session_id: return jsonify({"error": "缺少输入或 Session ID"}), 400
#         user_id = session['user_id']; chat_session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
#         if not chat_session: return jsonify({"error": "会话未找到或无权访问"}), 404

#         # --- 调用 RAG 核心 ---
#         generated_details = {}; raw_graph_strings = None; vector_retrieval = None
#         try:
#              # *** 你需要根据你的 Demo_chat 实现调整这里 ***
#              generated_details["vector_response"] = current_model.chat(user_input, mode='vector')
#              generated_details["graph_response"] = current_model.chat(user_input, mode='graph')
#              generated_details["hybrid_response"] = current_model.chat(user_input, mode='hybrid')
#              # *** 获取真实的检索结果 (需要你的类支持) ***
#              if hasattr(current_model, 'get_last_raw_graph_strings'): raw_graph_strings = current_model.get_last_raw_graph_strings()
#              if hasattr(current_model, 'get_last_retrieval_results'): vector_retrieval = current_model.get_last_retrieval_results('vector')
#              generated_details["vector_retrieval_result"] = vector_retrieval
#         except Exception as chat_err: return jsonify({"error": f"生成回答时核心模块出错：{chat_err}"}), 500

#         # --- 保存到数据库 ---
#         message_id_saved, timestamp_saved = None, None
#         try:
#             new_message = ChatMessage(
#                 session_id=session_id, query=user_input,
#                 vector_response=generated_details.get("vector_response"), graph_response=generated_details.get("graph_response"), hybrid_response=generated_details.get("hybrid_response"),
#                 vector_retrieval_json=json.dumps(vector_retrieval) if vector_retrieval is not None else None,
#                 graph_retrieval_raw=json.dumps(raw_graph_strings) if raw_graph_strings is not None else None, # 保存原始图谱字符串
#                 rag_mode_used=rag_mode )
#             db.session.add(new_message); db.session.commit()
#             message_id_saved, timestamp_saved = new_message.id, new_message.timestamp.isoformat() + 'Z'
#         except Exception as db_err: db.session.rollback(); return jsonify({"error": "保存历史记录时出错"}), 500

#         # --- 返回结果 ---
#         return jsonify({ "query": user_input, "vectorAnswer": generated_details.get("vector_response"), "graphAnswer": generated_details.get("graph_response"), "hybridAnswer": generated_details.get("hybrid_response"), "message_id": message_id_saved, "timestamp": timestamp_saved }), 200
#     except Exception as e: return jsonify({"error": f"处理生成请求时发生意外错误：{str(e)}"}), 500

# --- 格式化 SSE 输出 ---
def sse_pack(data_dict: dict) -> str:
    """将字典格式化为 Server-Sent Event (SSE) data 字段"""
    return f"data: {json.dumps(data_dict)}\n\n"

@app.route('/generate', methods=['POST'])
@login_required
def generate_answers():
    global current_model
    
    if not RAG_CORE_LOADED or current_model is None:
        return Response(sse_pack({"type": "error", "message": "模型未加载或当前不可用"}), mimetype='text/event-stream', status=501)
    if not request.is_json:
        return Response(sse_pack({"type": "error", "message": "请求必须是 JSON 格式"}), mimetype='text/event-stream', status=415)
    if not DB_INIT_SUCCESS:
        return Response(sse_pack({"type": "error", "message": "数据库服务当前不可用"}), mimetype='text/event-stream', status=503)

    try:
        data = request.json
        user_input = data.get("input")
        session_id_str = data.get("session_id")
        rag_mode_to_stream = data.get("rag_mode", "hybrid") # 客户端指定要流式获取的模式

        if rag_mode_to_stream not in ['vector', 'graph', 'hybrid']:
             return Response(sse_pack({"type": "error", "message": "无效的 RAG 模式请求"}), mimetype='text/event-stream', status=400)

        if not user_input or not session_id_str:
            return Response(sse_pack({"type": "error", "message": "缺少用户输入或会话 ID"}), mimetype='text/event-stream', status=400)

        user_id = session['user_id']
        try:
            session_id_int = int(session_id_str)
        except ValueError:
            return Response(sse_pack({"type": "error", "message": "无效的会话 ID 格式"}), mimetype='text/event-stream', status=400)
            
        chat_session = ChatSession.query.filter_by(id=session_id_int, user_id=user_id).first()
        if not chat_session:
            return Response(sse_pack({"type": "error", "message": "会话未找到或您无权访问此会话"}), mimetype='text/event-stream', status=404)

        # --- 流式生成器函数 ---
        def event_stream_generator():
            full_streamed_response_content = []
            # 这些变量用于存储各个RAG流程的检索结果，以便后续保存数据库
            # 您需要调整 Demo_chat 或其组件，使其在执行检索后能提供这些信息
            current_vector_retrieval_for_db = None
            current_graph_retrieval_for_db = None
            current_hybrid_context_for_db = None # 混合模式可能使用特定的组合上下文
            
            error_during_rag_call = None

            try:
                # **关键步骤：调用 Demo_chat 中经改造或新增的、支持流式的方法**
                # 下面的调用是假设性的，您需要根据 Demo_chat.py 的实际情况调整或实现：
                
                # 假设1: Demo_chat 类有一个统一的流式聊天方法
                # def stream_chat(self, message, mode, history=None) -> generator:
                #     # 1. 根据 mode 获取检索上下文 (vector_ctx, graph_ctx)
                #     # 2. 存储这些上下文到 self 的临时变量，供后续 get_last_retrieval_results 获取
                #     # 3. 构建 prompt
                #     # 4. yield from self.llm.chat_with_ai_stream(prompt, history)
                
                # 假设2: 或者，我们直接在 app.py 中编排，但更推荐封装在 Demo_chat 中
                # 为简化，我们假设 current_model 有一个改造后的 .chat() 方法，或者一个新的流式方法
                
                prompt_for_llm = ""
                history_for_llm = [] # 可选，从数据库加载历史

                # 1. 根据 rag_mode_to_stream 获取上下文并构建 Prompt
                #    这部分逻辑目前在 Demo_chat.py 的各个 chat 方法中，需要提取或改造
                if rag_mode_to_stream == "vector":
                    # 假设: current_model.chat_vector.prepare_streaming_context(user_input) 返回 (prompt, retrieval_data)
                    # 或者 current_model.chat_vector.get_retrieval_context()
                    # prompt_for_llm, current_vector_retrieval_for_db = current_model.chat_vector.some_method_to_get_prompt_and_retrieval(user_input)
                    # 为示意，我们直接调用其web_chat，但理想情况下web_chat应能返回流
                    # 这是一个很大的简化和假设，实际需要重构 ChatVectorRAG 等类
                    # ---- 概念性代码开始 ----
                    if hasattr(current_model.chat_vector, 'get_context_and_prompt'): # 理想的接口
                        prompt_for_llm, current_vector_retrieval_for_db = current_model.chat_vector.get_context_and_prompt(user_input, history_for_llm)
                    else: # 如果没有，需要您在 Demo_chat 或 ChatVectorRAG 中实现
                        # 模拟：获取上下文，然后构建prompt
                        # current_vector_retrieval_for_db = current_model.chat_vector.retrieval_result() # 这可能不正确，因为它可能是上次调用的结果
                        # prompt_for_llm = current_model.chat_vector._build_prompt(user_input, current_vector_retrieval_for_db, history_for_llm)
                        raise NotImplementedError("Demo_chat 或其组件需要提供获取RAG上下文和对应Prompt的方法以支持流式输出")
                    # ---- 概念性代码结束 ----

                elif rag_mode_to_stream == "graph":
                    # 类似地处理 graph RAG
                    # prompt_for_llm, current_graph_retrieval_for_db = current_model.chat_graph.some_method_to_get_prompt_and_retrieval(user_input)
                    raise NotImplementedError("Graph RAG 流式上下文和Prompt获取逻辑未实现")
                
                elif rag_mode_to_stream == "hybrid":
                    # hybrid 模式的上下文准备和 prompt 构建
                    # prompt_for_llm, current_hybrid_context_for_db = current_model.prepare_hybrid_prompt_and_retrieval(user_input)
                    raise NotImplementedError("Hybrid RAG 流式上下文和Prompt获取逻辑未实现")

                elif rag_mode_to_stream == "without_rag":
                    # ChatWithoutRAG = getattr(sys.modules.get('chat.chat_withoutrag'), 'ChatWithoutRAG', None) # 动态获取类
                    # if ChatWithoutRAG:
                    #     no_rag_chat_instance = ChatWithoutRAG(current_model.llm)
                    #     prompt_for_llm = no_rag_chat_instance._build_prompt(user_input, history_for_llm) # 假设有此方法
                    # else:
                    #     raise ImportError("ChatWithoutRAG 类未找到或未导入")
                    # 为简化，直接使用一个简单prompt
                    prompt_for_llm = user_input # 最简单的无RAG情况
                
                # 2. 调用底层的LLM流式接口
                print(f"向LLM发送流式请求 (模式: {rag_mode_to_stream}), Prompt: {prompt_for_llm[:100]}...")
                # 假设 self.llm 是 Demo_chat 中初始化的 LLM Client 实例
                answer_stream = current_model.llm.chat_with_ai_stream(prompt=prompt_for_llm, history=history_for_llm)

                for chunk_obj in answer_stream: # LLM Client 返回的原始 chunk 对象
                    # 解析 chunk_obj 以获取文本内容，这取决于您的 LLM Client 实现
                    # 例如，对于 OpenAI 兼容的客户端:
                    chunk_content = ""
                    if hasattr(chunk_obj, 'choices') and chunk_obj.choices:
                        delta = chunk_obj.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            chunk_content = delta.content
                    
                    if chunk_content:
                        full_streamed_response_content.append(chunk_content)
                        yield sse_pack({"type": "chunk", "content": chunk_content})
                        # time.sleep(0.01) 

            except NotImplementedError as nie: # 捕获我们自己抛出的未实现错误
                error_during_rag_call = f"功能实现中: {str(nie)}"
                print(error_during_rag_call)
                yield sse_pack({"type": "error", "message": error_during_rag_call})
            except Exception as e:
                error_during_rag_call = f"生成回答时核心模块出错: {str(e)}"
                print(error_during_rag_call)
                traceback.print_exc()
                yield sse_pack({"type": "error", "message": "处理您的请求时发生内部错误。"})
            
            # --- 流结束 ---
            if error_during_rag_call:
                yield sse_pack({"type": "end", "status": "error_rag_core", "message": error_during_rag_call})
                return

            final_streamed_answer = "".join(full_streamed_response_content)
            print(f"模式 '{rag_mode_to_stream}' 流式回答完成。")

            # --- [重要] 获取用于数据库保存的完整信息 ---
            # 此时，final_streamed_answer 是流式模式的完整答案。
            # 我们还需要其他模式的答案以及所有相关的检索信息才能完整保存 ChatMessage。
            
            # (此部分逻辑与上一轮回复中的非流式 /generate 类似，在流结束后获取)
            all_responses_for_db = { "vector_response": None, "graph_response": None, "hybrid_response": None }
            all_responses_for_db[f"{rag_mode_to_stream}_response"] = final_streamed_answer

            # 同步获取其他模式的答案（如果需要）
            # ... (省略这部分代码，参考上一轮回复中的实现，它会调用 current_model.chat(stream=False))
            # ... (以及获取 current_vector_retrieval_for_db, current_graph_retrieval_for_db 的逻辑)
            # 为确保能运行，暂时将其他模式的回答和检索结果设为占位符或从流式模式的结果推断
            # 您需要根据实际需求完善这里
            if rag_mode_to_stream != "vector":
                 all_responses_for_db["vector_response"] = "同步获取vector答案（待实现）"
            if rag_mode_to_stream != "graph":
                 all_responses_for_db["graph_response"] = "同步获取graph答案（待实现）"
            if rag_mode_to_stream != "hybrid":
                 all_responses_for_db["hybrid_response"] = "同步获取hybrid答案（待实现）"
            
            # 假设检索结果可以通过某种方式获取，这里用占位符
            # current_vector_retrieval_for_db = current_model.get_last_retrieval_results('vector') if hasattr(current_model, 'get_last_retrieval_results') else None
            # current_graph_retrieval_for_db = current_model.get_last_raw_graph_strings() if hasattr(current_model, 'get_last_raw_graph_strings') else None


            # --- 将交互结果保存到数据库 ---
            message_id_saved = None; timestamp_saved_iso = None; db_save_error_message = None
            try:
                print("准备将聊天记录保存到数据库...")
                new_message = ChatMessage(
                    session_id=session_id_int, query=user_input,
                    vector_response=all_responses_for_db.get("vector_response"),
                    graph_response=all_responses_for_db.get("graph_response"),
                    hybrid_response=all_responses_for_db.get("hybrid_response"),
                    vector_retrieval_json=json.dumps(current_vector_retrieval_for_db) if current_vector_retrieval_for_db else None,
                    graph_retrieval_raw=json.dumps(current_graph_retrieval_for_db) if current_graph_retrieval_for_db else None,
                    rag_mode_used=rag_mode_to_stream
                )
                db.session.add(new_message)
                db.session.commit()
                message_id_saved = new_message.id
                timestamp_saved_iso = new_message.timestamp.isoformat() + 'Z'
                print(f"聊天记录 {message_id_saved} 已成功保存。")
            except Exception as db_err:
                db.session.rollback()
                db_save_error_message = "未能成功将此条聊天记录保存到数据库。"
                print(f"数据库保存错误: {db_err}"); traceback.print_exc()

            final_sse_payload = {"type": "end", "status": "success" if not db_save_error_message else "warning_dbsave_failed",
                                 "message_id": message_id_saved, "timestamp": timestamp_saved_iso}
            if db_save_error_message: final_sse_payload["db_error"] = db_save_error_message
            yield sse_pack(final_sse_payload)
            print("流式响应和数据库操作完成。")

        return Response(event_stream_generator(), mimetype='text/event-stream')

    except Exception as setup_e:
        error_message = f"处理 /generate 请求时发生意外错误 (流开始前): {str(setup_e)}"
        print(error_message); traceback.print_exc()
        def error_stream_response(): yield sse_pack({"type": "error", "message": f"处理请求时发生严重错误: {str(setup_e)}"})
        return Response(error_stream_response(), mimetype='text/event-stream', status=500)

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
# --- 检索详情 API ---
@app.route('/get-vector/<item_id>', methods=['GET'])
@login_required
def get_vector(item_id):
    # 获取与 item_id 相关的 vector 数据
    session_name = request.args.get("sessionName")
    dataset_name = request.args.get("datasetName")


    session_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name,f"{session_name}.json")
    )
    filtered_data = load_and_filter_data(session_file, item_id)
    retrieval_result = []
    if filtered_data:
         # 确保前端期望的 'chunks' 键存在
        if 'vector_retrieval_result' in filtered_data and isinstance(filtered_data['vector_retrieval_result'], list):
            
             # 简单地将 retrieve_results 的值（假设是文本列表）转换为 chunk 对象
             for text_list in filtered_data['vector_retrieval_result']:
                 retrieval_result.append(text_list)
        
        # print("#######retrieval_result########",retrieval_result)
        result = {
            'id': item_id,
            'chunks': retrieval_result
            
        }

        return jsonify(result)  # 返回处理后的数据
    else:
        return jsonify({'error': f'Item not found or invalid ID format for vector lookup: {item_id}'}), 404

@app.route('/get-graph/<item_id>', methods=['GET'])
@login_required
def get_graph(item_id):
    session_name = request.args.get("sessionName")
    dataset_name = request.args.get("datasetName")


    session_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name,f"{session_name}.json")
    )
    filtered_data = load_and_filter_data(session_file, item_id)
    if filtered_data and 'graph_retrieval_result' in filtered_data:
        evidence_entity,evidence_path = get_evidence(EVIDENCE_FILE_PATH,item_id)
        # 转换 retrieve_results 为三元组
        triples = convert_rel_to_triplets(filtered_data["graph_retrieval_result"])
        print("############triples###########",triples)
        if not triples:
             # print(f"Warning: No triples generated for item {item_id}. Returning empty graph.")
             return jsonify({'edges': [], 'nodes': [], 'highlighted-edge': [], 'highlighted-node': []})

        json_result = triples_to_json(triples,evidence_entity,evidence_path)
        # print("================= GRAPH RESPONSE =================") # 原始调试信息
        # print(json.dumps(json_result, indent=2))                  # 原始调试信息
        # print("================================================") # 原始调试信息
        return jsonify(json_result)  # 返回找到的数据
    elif filtered_data is None:
         # 如果 load_and_filter_data 返回 None (ID 不是整数或文件找不到)
        return jsonify({'error': f'Item not found or invalid ID format for graph lookup: {item_id}'}), 404
    else:
        # 如果找到了数据但格式不对
        print(f"Warning: Graph data format error or missing 'retrieve_results' for item {item_id}")
        return jsonify({'error': f'Data format error for graph item {item_id}'}), 500

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
    """根据 models.py 创建数据库表。"""
    if DB_INIT_SUCCESS: # 检查数据库是否成功初始化
        with app.app_context():
            try: 
                db.create_all()
                print("数据库表创建成功（或已存在）。")
            except Exception as e: print(f"错误：创建数据库表时出错：{e}"); traceback.print_exc()
    else: print("错误：无法创建表，数据库初始化失败或未配置。")

# --- 主程序入口 ---
if __name__ == '__main__':
    ui_port = int(os.environ.get('FLASK_PORT', 5000))
    print(f" * 启动 Flask 应用于 http://0.0.0.0:{ui_port}")
    # (省略启动信息打印 - 同前)
    if not RAG_CORE_LOADED: print(" * 警告：RAG 核心未加载。")
    if not DB_INIT_SUCCESS: print(" * 错误：数据库未正确配置或初始化失败，功能受限。")
    print("#################################")
    app.run(host='0.0.0.0', port=ui_port, debug=False, threaded=True)