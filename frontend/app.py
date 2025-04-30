import json
import re
import time
import uuid
from datetime import datetime, timezone
from config.config import Config
from flask import Flask, Response, request, jsonify, render_template, session
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask_cors import CORS
from zhipuai import ZhipuAI
from user import User
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

app = Flask(__name__)
app.secret_key = 'ac1e22dfb44b87ef38f5bf2cd1cb0c6f93bb0a67f1b2d8f7'  # 用于 flash 消息
CORS(app) # Enable CORS for all routes

current_model = None
def read_json_lines(file_path):
    """
    逐行读取 JSON 文件，并返回所有解析的记录。

    :param file_path: 要读取的 JSON 文件路径
    :return: 解析后的记录列表
    """
    items = []  # 用于存储所有解析的记录
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 解析每一行 JSON
                record = json.loads(line.strip())
                # 将解析的记录添加到列表
                items.append(record)
            except json.JSONDecodeError as e:
                # 如果遇到解析错误，输出错误信息
                print(f"解析错误: {e} - 无法解析行: {line}")
    
    return items  # 返回所有解析的记录列表

# --- 新增: 用于模拟会话和历史记录的内存存储 ---
# 警告: 数据将在服务器重启时丢失! 实际应用需要数据库。
sessions_storage = {} # 存储格式: { "session_id": {"id": "...", "name": "..."} }
history_storage = {} # 存储格式: { "session_id": [ { "id": "item_id", ... }, ... ] }

# 初始化一个默认会话 (用于模拟)
default_session_id = str(uuid.uuid4())
sessions_storage[default_session_id] = {"id": default_session_id, "name": "Default Session"}
history_storage[default_session_id] = []
# --- 结束内存存储 ---


# --- 原始路由 ---
@app.route('/')
def index():
    return render_template('demo.html')  # 大模型页面
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/register')
def register():
    return render_template('register.html')  # 注册页面

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

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

def convert_to_triples(retrieve_results):
    """
    将 retrieve_results 中的字符串转换为三元组形式，支持多种边的关系。
    """
    triples = set()
    if not isinstance(retrieve_results, dict): # 增加类型检查
        # print(f"Warning: retrieve_results is not a dict ({type(retrieve_results)}), skipping triple conversion.")
        return list(triples)

    for key, value_list in retrieve_results.items():
        if not isinstance(value_list, list): continue # 跳过非列表的值

        for value in value_list:
            if not isinstance(value, str): continue # 只处理字符串
            # 使用 parse_relationship 解析关系 (使用 split_relation)
            parsed_triples = split_relation(value)
            # 将解析出的三元组加入到结果中
            for t in parsed_triples:
                if len(t) == 3: # 确保是有效三元组
                    triples.add(t)
    return list(triples)


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

# --- 原始 API 接口 ---

@app.route('/get-graph/<item_id>', methods=['GET'])
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



# @app.route('/get-dataset', methods=['POST'])
# def get_dataset():
#     data = request.get_json()
#     hop = data.get('hop')
#     type = data.get('type')
#     entity = data.get('entity')
#     dataset = data.get('dataset')
#     # {'hop': 'single_hop', 'type': 'specific', 'entity': 'single_entity', 'dataset': 'rgb'}
#     print(data,hop,type,entity,dataset)
#     relative_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"
    # # if os.path.exists(relative_path):
    #     print("############存在#############")






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










@app.route('/get-vector/<item_id>', methods=['GET'])
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

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    phone = data.get('phone')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not all([username, email, phone, password, confirm_password]):
        return jsonify({"error": "所有字段都是必需的！"}), 400
    try:
        user = User(username=username, email=email, phone=phone, password=password)
        if user.register(confirm_password):
            return jsonify({"message": "注册成功！"}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e: # 捕获其他可能的错误
         print(f"Registration error: {e}")
         traceback.print_exc()
         return jsonify({"error": "注册过程中发生内部错误。"}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not all([username, password]):
        return jsonify({"error": "用户名和密码是必需的！"}), 400
    try:
        user = User(username=username, email=None, phone=None, password=password)
        if user.login(password):
            session['username'] = username
            session['user_id'] = user.get_user_id() # 假设 User 类有 get_user_id 方法
            return jsonify({"message": "登录成功！"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
         print(f"Login error: {e}")
         traceback.print_exc()
         return jsonify({"error": "登录过程中发生内部错误。"}), 500

# 原始 /api/chat 接口 (流式，可能用于直接测试 LLMRAGEnv)
@app.route('/api/chat', methods=['POST'])
def web_chat():
    try: # 封装以捕获初始化错误
        chat_env = LLMRAGEnv() # 每次请求创建新实例?
        data = request.get_json()
        user_input = data.get('user_input')
        # 从请求中获取参数，提供默认值
        model = data.get("model", "qwen:0.5b") # 原始默认值
        rag_mode = data.get('mode', "vector rag") # 原始默认值 (注意空格)
        graph_db = data.get("graph_db", "nebulagraph")
        vector_db = data.get("vector_db", "milvus")
        history = data.get("history", []) # 允许从请求传递历史

        if not user_input:
             return jsonify({"error": "user_input is required"}), 400

        # 使用生成器进行流式响应
        def generate_chat():
            try:
                answer_stream = chat_env.web_chat(
                    message=user_input, history=history,
                    op0=model, op1=rag_mode, op2=graph_db, op3=vector_db
                )
                for msg in answer_stream:
                    # 按 SSE 格式发送 JSON 消息
                    # print(f"data: {json.dumps({'message': msg})}") # 原始调试
                    yield f"data: {json.dumps({'message': msg})}\n\n"
                    time.sleep(0.05) # 稍微降低延迟

                # 流结束时，发送特殊的结束标志
                yield "data: {\"message\": \"[END]\"}\n\n"

            except Exception as e:
                # 如果发生异常，返回错误消息
                print(f"Error occurred during stream generation: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # 返回流式数据
        return Response(generate_chat(), content_type='text/event-stream')
    except Exception as e:
         # 处理 LLMRAGEnv 初始化等错误
         error_msg = f"Failed to initialize or run chat: {e}"
         print(error_msg)
         traceback.print_exc()
         # 不能在流式响应启动前返回 jsonify，所以这里可能无法很好地报告错误
         # 或许应该返回一个包含错误的 SSE 事件
         def error_stream():
              yield f"data: {json.dumps({'error': error_msg})}\n\n"
         return Response(error_stream(), content_type='text/event-stream', status=500)

# 原始 /display_generate 接口 (似乎是用于测试或特定显示?)
@app.route('/display_generate', methods=['GET'])
def display_generate():
    item_id = request.args.get("item_id")
    option_value = request.args.get("option_value")

    if not item_id or not option_value:
        return jsonify({"error": "Missing parameters"}), 400

    # 这里模拟数据读取 (原始逻辑)
    result = {
        "query": f"Query for item {item_id} with {option_value}",
        "answer": f"Generated answer for {option_value} on item {item_id}"
    }
    return jsonify({"result": result})

# --- 新增: /generate 接口 (供前端 Send 按钮调用) ---
@app.route('/generate', methods=['POST'])
def generate_answers():
    global current_model
    if current_model is None:
        return jsonify({"error": "Model not loaded. Please apply settings first."}), 400

    try:
        data = request.json
        user_input = data.get("input")
        if not user_input:
            return jsonify({"error": "Input query is missing"}), 400

        # --- 实际调用 Demo_chat 获取答案 ---
        # 假设 Demo_chat 实例有 chat 方法，并接受 mode 参数
        simulated_vector_answer = "Vector answer placeholder"
        simulated_graph_answer = "Graph answer placeholder"
        simulated_hybrid_answer = "Hybrid answer placeholder"

        try:
            # 尝试调用 chat 方法获取不同模式的答案
            # 需要 Demo_chat 类支持这种调用方式
            simulated_vector_answer = current_model.chat(user_input, mode='vector')
            simulated_graph_answer = current_model.chat(user_input, mode='graph')
            simulated_hybrid_answer = current_model.chat(user_input, mode='hybrid')
        except AttributeError:
            print("Warning: current_model.chat() may not support 'mode' parameter. Using chat_test() as fallback.")
            # 如果 chat 不支持 mode，尝试调用 chat_test 或其他方法
            try:
                 # 假设 chat_test 返回的是当前模型配置下的答案
                 primary_answer = current_model.chat_test(user_input) # 或者 current_model.chat(user_input)
                 # 根据 current_model 的内部状态（如果可访问）或默认值填充
                 simulated_vector_answer = primary_answer # 示例：都用主答案填充
                 simulated_graph_answer = primary_answer
                 simulated_hybrid_answer = primary_answer
            except Exception as fallback_e:
                 print(f"Error during fallback chat call: {fallback_e}")
                 # 保持占位符或设置错误消息
                 simulated_vector_answer = f"Error in fallback: {fallback_e}"
                 simulated_graph_answer = f"Error in fallback: {fallback_e}"
                 simulated_hybrid_answer = f"Error in fallback: {fallback_e}"

        except Exception as e:
            print(f"Error calling current_model.chat(): {e}")
            traceback.print_exc()
            # 设置错误信息给所有答案
            error_msg = f"Error generating: {e}"
            simulated_vector_answer = error_msg
            simulated_graph_answer = error_msg
            simulated_hybrid_answer = error_msg
        # --- 结束调用 ---

        response_data = {
            "query": user_input, # 回显查询
            "vectorAnswer": simulated_vector_answer,
            "graphAnswer": simulated_graph_answer,
            "hybridAnswer": simulated_hybrid_answer,
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error in /generate endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500


# --- 用户认证相关接口 (原始) ---
@app.route('/api/get-username', methods=['GET'])
def get_username():
    username = session.get('username')
    if username:
        return jsonify({"username": username}), 200
    else:
        return jsonify({"error": "用户未登录"}), 401 # 使用 401 Unauthorized 更合适

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "注销成功"}), 200


# @app.route('/get-dataset', methods=['POST'])
# def get_dataset():
#     data = request.get_json()
#     hop = data.get('hop')
#     type = data.get('type')
#     entity = data.get('entity')
#     dataset = data.get('dataset')
#     # {'hop': 'single_hop', 'type': 'specific', 'entity': 'single_entity', 'dataset': 'rgb'}
#     print(data,hop,type,entity,dataset)
#     relative_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"
    # # if os.path.exists(relative_path):
    #     print("############存在#############")



# --- 模型加载接口 (原始) ---
# @app.route('/load_model', methods=['POST'])
# def load_model():
#     global current_model
#     try:
#         data = request.json
#         model_name = data.get("model_name")
#         url = data.get("url") # url 在 Demo_chat 中似乎未使用 (原始注释)
#         key = data.get("key")
#         dataset_info = data.get("dataset") # 获取数据集名称
#         hop = dataset_info.get('hop')
#         type = dataset_info.get('type')
#         entity = dataset_info.get('entity')
#         dataset = dataset_info.get('dataset')
#         session = dataset_info.get('session')
#         dataset_path = f"../data/{hop}/{type}/{entity}/{dataset}/{dataset}.json"    
#         if key == "" or key is None: # 处理空或 None 的 key
#             key = "ollama" # 默认 key
#         print(f"Received /load_model: model={model_name}, key={'<default_ollama>' if key=='ollama' else '<provided>'}, dataset={dataset}") # 修正日志

#         if not model_name: return jsonify({"status": "error", "message": "缺少模型名称"}), 400
#         if not dataset:
#             print("Warning: Dataset parameter is missing in /load_model request.")
#             # 根据需要决定是否强制要求数据集
#             # return jsonify({"status": "error", "message": "缺少数据集参数"}), 400

#         # 关闭旧模型 (原始逻辑，简单替换引用)
#         if current_model is not None:
#             print(f"正在替换现有模型实例。")
#             current_model = None # 允许垃圾回收

#         # 加载新模型
#         print(f"正在加载模型: {model_name} (API Key: {'*'*(len(key)-3)+key[-3:] if key != 'ollama' and key else 'ollama'}, 数据集: {dataset})")
#         current_model = Demo_chat(model_name=model_name, api_key=key, dataset_name=dataset,dataset_path=dataset_path,path_name=session) # 传递数据集参数

#         # 测试模型 (原始逻辑)
#         test_result = current_model.chat_test() # 假设 chat_test 不需要输入
#         print(f"模型测试结果: {test_result}")
        
#         def generate():
#             # 发送初始状态
#             yield json.dumps({"status": "start", "message": f"模型 {model_name} (数据集: {dataset}) 加载成功"}) + "\n"
            
#             # 处理每个项目并立即发送
#             for item_data in current_model.new_history_chat():
#                 yield json.dumps({"status": "processing", "item_data": item_data}) + "\n"
            
#             # 发送完成状态
#             yield json.dumps({"status": "complete", "message": "所有项目处理完成"}) + "\n"

#         return Response(generate(), mimetype='text/event-stream')

#     except Exception as e:
#         print(f"加载模型时出错: {e}")
#         traceback.print_exc() # 打印完整错误堆栈
#         return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/load_model', methods=['POST'])
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





# --- 新增: 会话和历史记录 API 接口 (使用内存模拟) ---

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """(模拟) 返回会话列表。"""
    session_list = list(sessions_storage.values())
    return jsonify(session_list)

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """(模拟) 创建新会话。"""
    data = request.get_json()
    session_name = data.get('name')
    if not session_name: return jsonify({"error": "需要会话名称"}), 400
    # 可选: 检查名称冲突
    # if any(s['name'] == session_name for s in sessions_storage.values()): return jsonify({"error": "会话名称已存在"}), 409

    new_id = str(uuid.uuid4())
    new_session = {"id": new_id, "name": session_name}
    sessions_storage[new_id] = new_session
    history_storage[new_id] = []
    print(f"(模拟) 已创建会话: {new_session}")
    return jsonify(new_session), 201

# @app.route('/api/sessions/<sessionId>/history', methods=['GET'])
# def get_session_history(sessionId):
#     """(模拟) 返回指定会话的历史记录。"""
#     if sessionId not in history_storage: return jsonify({"error": "会话未找到"}), 404
#     # 按时间戳降序返回 (可选)
#     session_history = sorted(history_storage[sessionId], key=lambda item: item.get('timestamp', ''), reverse=True)
#     return jsonify(session_history)

@app.route('/api/sessions/<sessionId>/history', methods=['POST'])
def add_history_item(sessionId):
    """(模拟) 向指定会话添加历史记录项。"""
    if sessionId not in sessions_storage: return jsonify({"error": "会话未找到"}), 404

    data = request.get_json()
    query = data.get('query')
    answer = data.get('answer')
    type = data.get('type', 'INFO')
    details = data.get('details', {}) # 包含 vectorAnswer 等
    if not query: return jsonify({"error": "需要查询内容"}), 400

    item_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    new_item = {
        "id": item_id, "sessionId": sessionId, "query": query, "answer": answer,
        "type": type, "details": details, "timestamp": timestamp
    }
    history_storage[sessionId].insert(0, new_item) # 插入到开头 (最新)
    print(f"(模拟) 已添加历史项 {item_id} 到会话 {sessionId}")
    return jsonify(new_item), 201

@app.route('/list-history', methods=['POST'])  # 改为 POST，接收 JSON 数据
def list_history_files():
    # 解析前端传来的 JSON 数据
    data = request.get_json()
    dataset_name = data.get("selectedDatasetName")
    
    print(f"收到请求，数据集名: {dataset_name}")  # 调试日志

    if not dataset_name:
        return jsonify({"files": [], "error": "Missing dataset name"}), 400

    # 获取目标 chat_history 子目录路径
    history_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name)
    )
    print(f"历史目录路径: {history_dir}")  # 调试日志

    if not os.path.exists(history_dir):
        print("目录不存在")  # 调试日志
        return jsonify({"files": []})

    # 获取该数据集下的所有 JSON 文件名（不含后缀）
    files = [
        os.path.splitext(name)[0]
        for name in os.listdir(history_dir)
        if os.path.isfile(os.path.join(history_dir, name)) and name.endswith(".json")
    ]
    
    print(f"找到文件: {files}")  # 调试日志
    return jsonify({"files": files})

@app.route('/create-history-session', methods=['POST'])
def create_history_session(): 
    data = request.get_json()
    session_name = data.get("sessionName")
    dataset_name = data.get("datasetName")


    session_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name,f"{session_name}.json")
    )
    try:
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        with open(session_file, 'w', encoding='utf-8') as f:
            f.close()
        return jsonify({"success": True, "sessionName": session_name, "path": session_file})
    except Exception as e:
        # 发生错误时，返回失败响应及错误信息
        return jsonify({"success": False, "message": str(e)}), 500




    
@app.route('/api/sessions/history', methods=['GET'])
def get_session_history():
    # 获取查询参数 dataset 和 session
    dataset_name = request.args.get('dataset')
    session_name = request.args.get('session')
    if not dataset_name or not session_name:
        return jsonify({"success": False, "message": "数据集名称或会话名称未提供"}), 400
    session_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'backend/llmragenv','chat_history', dataset_name,f"{session_name}.json")
    )
    if not os.path.exists(session_file):
        return jsonify({"success": False, "message": f"历史记录文件 {session_name}.json 不存在"}), 404
    try:
        history_data = read_json_lines(session_file) 
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500




if __name__ == '__main__':
    # use_reloader=False 对于使用全局变量进行内存存储很重要
    app.run(host='0.0.0.0', port=int(Config.get_instance().get_with_nested_params("server", "ui_port")), debug=True, use_reloader=False)