import json
import re
import time
from flask import Flask, Response, request, jsonify, render_template, session
# from flask_socketio import SocketIO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask_cors import CORS  # Import CORS
from zhipuai import ZhipuAI
from user import User  # 假设你的 User 类定义在 user.py 中
from llmragenv.llmrag_env import LLMRAGEnv
from evaluator import simulate  # Import simulate module
from llmragenv.demo_chat import Demo_chat
import threading

app = Flask(__name__)
app.secret_key = 'ac1e22dfb44b87ef38f5bf2cd1cb0c6f93bb0a67f1b2d8f7'  # 用于 flash 消息
CORS(app) # Enable CORS for all routes - important for frontend to access backend from different origins


current_model = None

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

#测试路径

VECTOR_FILE_PATH = '/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/vectorrag/analysis_retrieval___top5_2024-11-26_21-32-23.json'
GRAPH_FILE_PATH = '/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_retrieval_merged.json'
EVIDENCE_FILE_PATH = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb_evidence.json"


# [Rest of your functions: load_and_filter_data, find_right_arrow, find_left_arrow, get_all_dash, find_dash_positions, split_relation, convert_to_triples, triples_to_json, get_evidence, get_graph, read_file, get_vector, adviser, web_chat, test_chat, get_username, logout, register_user, login_user remain the same]
# ... (Paste all your functions here: load_and_filter_data, find_right_arrow, find_left_arrow, get_all_dash, find_dash_positions, split_relation, convert_to_triples, triples_to_json, get_evidence, get_graph, read_file, get_vector, adviser, web_chat, test_chat, get_username, logout, register_user, login_user) ...
# [
    # {
    #     "id": 83,
    #     "question": "Who won the 2022 Tour de France?",
    #     "answer": [
    #         "Jonas Vingegaard"
    #     ],
    #     "vector_response": "Denmark's Jonas Vingegaard (Jumbo-Visma) won the yellow jersey as the overall winner of the 2022 Tour de France.",
    #     "graph_response": "Jonas Vingegaard",
    #     "hybrid_response": "Jonas Vingegaard",
    #     "type": "GREEN"
    # },
##加载响应的id数据
def load_and_filter_data(file_path, item_id):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # 加载 JSON 数据
            # 通过 item_id 查找对应的元素
            filtered_data = next((item for item in data if item.get('id') == int(item_id)), None)
            return filtered_data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}.")
        return None

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
        if s[i-1] == " "  or s[i+1] == " ":
            dash_positions.append(i)
        

    return dash_positions




def split_relation(rel_seq):
    parts = []
    all_dash = get_all_dash(rel_seq)
    right_arrows = find_right_arrow(rel_seq)
    left_arrows = find_left_arrow(rel_seq)
    dash_positions = find_dash_positions(rel_seq,all_dash)

    arrows_index = sorted(right_arrows+left_arrows)

    if len(arrows_index) == 1:
        if arrows_index[0] in right_arrows:
            source = rel_seq[:dash_positions[0]]
            rel = rel_seq[dash_positions[0]+1:arrows_index[0]]
            dst = rel_seq[arrows_index[0]+2:]
            parts.append((source,rel,dst))
        else:
            dst = rel_seq[:arrows_index[0]]
            rel = rel_seq[arrows_index[0]+2:dash_positions[0]]
            source = rel_seq[dash_positions[0]+1:]
            parts.append((source,rel,dst))

        return parts

    ###多跳的分解###
    i = 0
    for arrows in arrows_index:
        if  arrows in right_arrows:
            if i == 0:
                source = rel_seq[:dash_positions[0]].strip()
                rel = rel_seq[dash_positions[0]+1:arrows_index[0]].strip()
                dst = rel_seq[arrows_index[0]+2:min(dash_positions[1],arrows_index[1])].strip()
                parts.append((source,rel,dst))
                i+=1
            elif i == len(arrows_index)-1:
                dst = rel_seq[arrows_index[-1]+2:].strip()
                rel = rel_seq[dash_positions[-1]+1:arrows_index[-1]].strip()
                source = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):dash_positions[-1]].strip()
                parts.append((source,rel,dst))
                i+=1

            else:#既不是第一个也不是最后一个
                source = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):dash_positions[i]].strip()
                rel = rel_seq[dash_positions[i]+1:arrows_index[i]].strip()
                dst = rel_seq[arrows_index[i]+2:min(dash_positions[i+1],arrows_index[i+1])].strip()
                parts.append((source,rel,dst))
                i+=1


        if arrows in left_arrows:
            if i == 0:
                dst = rel_seq[:arrows_index[i]].strip()
                rel = rel_seq[arrows_index[i]+2:dash_positions[i]].strip()
                source = rel_seq[dash_positions[i]+1:min(dash_positions[i+1],arrows_index[i+1])].strip()
                parts.append((source,rel,dst))
                i+=1
            elif i == len(arrows_index)-1:
                source = rel_seq[dash_positions[i]+1:].strip()
                rel = rel_seq[arrows_index[i]+2:dash_positions[i]].strip()
                dst = rel_seq[max(arrows_index[i-1]+2,dash_positions[i-1]+1):arrows_index[i]].strip()
                parts.append((source,rel,dst))
                i+=1
                
            else:
                source = rel_seq[dash_positions[i]:min(dash_positions[i+1],arrows_index[i+1])].strip()
                rel = rel_seq[arrows_index[i]+2:dash_positions[i]].strip()
                dst = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):arrows_index[i]].strip()
                parts.append((source,rel,dst))
                i+=1


    return parts





def convert_to_triples(retrieve_results):
    """
    将 retrieve_results 中的字符串转换为三元组形式，支持多种边的关系。
    """
    triples = set()
    
    for key, value_list in retrieve_results.items():
        
        for value in value_list:
            # 使用 parse_relationship 解析关系
            parsed_triples = split_relation(value)
            
            # 将解析出的三元组加入到结果中
            for t in parsed_triples:
                triples.add(t)
                
    return list(triples)
            

def triples_to_json(triples,evdience_entity,evdience_path):

    colors = [
        "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
        "#B5EAD7", "#ECC5FB", "#FFC3A0", "#FF9AA2", "#FFDAC1",
        "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2",
        "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA",
        "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1",
        "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2",
        "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA",
        "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1",
        "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2",
        "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA",
        "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1"
    ]

    json_result = {'edges': [], 'nodes': [],'highlighted-edge':[],'highlighted-node':[]}
    node_set = set()  # 用于追踪已经添加的节点

    import random
    print(f"Triples:{triples}")

    for triple in triples:
        source = triple[0]
        relationship = triple[1]
        destination = triple[2]

        # 添加边
        if relationship in evdience_path:
            json_result['highlighted-edge'].append({
            'data': {
                'label': relationship,
                'source': source,
                'target': destination,
                'color': colors[random.randint(0,54)] # 可以根据需要自定义颜色
            }
        })
            json_result['edges'].append({
                'data': {
                    'label': relationship,
                    'source': source,
                    'target': destination,
                    'color': colors[random.randint(0,54)] # 可以根据需要自定义颜色
                }
            })
        else:

            json_result['edges'].append({
                'data': {
                    'label': relationship,
                    'source': source,
                    'target': destination,
                    'color': colors[random.randint(0,54)] # 可以根据需要自定义颜色
                }
            })

        # 添加源节点和目标节点（避免重复）
        if source in evdience_entity and source not in node_set:
            json_result['highlighted-node'].append({
                'data': {
                    'id': source,
                    'label': source,
                    'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                }
            })
            json_result['nodes'].append({
                    'data': {
                        'id': source,
                        'label': source,
                        'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                    }
                })
            node_set.add(source)
        else:
            if source not in node_set:
                json_result['nodes'].append({
                    'data': {
                        'id': source,
                        'label': source,
                        'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                    }
                })
                node_set.add(source)
            
        if destination in evdience_entity and destination not in node_set:
            json_result['highlighted-node'].append({
                'data': {
                    'id': destination,
                    'label': destination,
                    'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                }
            })
            json_result['nodes'].append({
                    'data': {
                        'id': destination,
                        'label': destination,
                        'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                    }
                })
            node_set.add(destination)
        else:    
            if destination not in node_set:
                json_result['nodes'].append({
                    'data': {
                        'id': destination,
                        'label': destination,
                        'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                    }
                })
                node_set.add(destination)

    return json_result


def get_evidence(file_path,item_id):
    with open(file_path, 'r') as file:
        data = json.load(file)  # 加载 JSON 数据
            # 通过 item_id 查找对应的元素
        e = next((item for item in data if item.get('id') == int(item_id)), None)

        entity = []
        path = []
        evidence_triples = e["merged_triplets"]
        
        # 遍历 evidence 中的每个三元组，获取实体
        for t in evidence_triples:
            for i in t:
                if len(i) == 3:
                    entity.append(i[0])  # 实体的第一个元素
                    path.append(i[1])
                    entity.append(i[2])  # 实体的第三个元素
        
    return entity,path
    

@app.route('/get-graph/<item_id>', methods=['GET'])
def get_graph(item_id):
    # 获取与 item_id 相关的 graph 数据
    filtered_data = load_and_filter_data(GRAPH_FILE_PATH, item_id)
    evidence_entity,evidence_path = get_evidence(EVIDENCE_FILE_PATH,item_id)
    if filtered_data:
        # 转换 retrieve_results 为三元组
        filtered_data = convert_to_triples(filtered_data['retrieve_results'])
        json_result = triples_to_json(filtered_data,evidence_entity,evidence_path)
        print("============================================")
        print(json_result)
        print("============================================")

        return jsonify(json_result)  # 返回找到的数据
    else:
        return jsonify({'error': 'Item not found'}), 404



@app.route('/read-file', methods=['GET'])
def read_file():
    try:
        # 设定文件路径
        file_path = '/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/test.json'
        
        # 读取文件并解析 JSON 内容
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 这里可以根据需要处理数据
        # 例如，如果你只想返回某些字段，可以在这里做处理
        result = []
        for item in data:
            # 假设你想返回 id, question, answer 和 response 字段 
            result.append({
                'id': item.get('id'),
                'question': item.get('question'),
                'answer': item.get('answer'),
                'hybrid_response': item.get('hybrid_response'),
                'type' : item.get('type')
            })

        # 返回经过处理的数据
        return jsonify({'content': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-vector/<item_id>', methods=['GET'])
def get_vector(item_id):
    # 获取与 item_id 相关的 vector 数据
    filtered_data = load_and_filter_data(VECTOR_FILE_PATH, item_id)
    if filtered_data:
        return jsonify(filtered_data)  # 返回找到的数据
    else:
        return jsonify({'error': 'Item not found'}), 404





@app.route('/get_suggestions', methods=['GET'])
def adviser():
    # 假设下面这些文件路径是正确的
    # /home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_generation___merged.json
    rgb_graph_generation = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_generation___merged.json"
    rgb_graph_retrieval = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/graphrag/analysis_retrieval_merged.json"
    rgb_vector_generation = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/vectorrag/analysis_generation___top5_2024-11-26_21-32-23.json"
    rgb_vector_retrieval = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb/vectorrag/analysis_retrieval___top5_2024-11-26_21-32-23.json"

    # 假设 statistic_error_cause 函数已经定义
    v_retrieve_error, v_lose_error, v_lose_correct =  simulate.statistic_error_cause(rgb_vector_generation, rgb_vector_retrieval, "vector")
    g_retrieve_error, g_lose_error, g_lose_correct = simulate.statistic_error_cause(rgb_graph_generation, rgb_graph_retrieval, "graph")

    suggestions = {
        "vector_retrieve_error": v_retrieve_error,
        "vector_lose_error": v_lose_error,
        "vector_lose_correct": v_lose_correct,
        "graph_retrieve_error": g_retrieve_error,
        "graph_lose_error": g_lose_error,
        "graph_lose_correct": g_lose_correct,
        "advice": "这里是对用户的建议"
    }

    print(suggestions)

    # 返回 JSON 格式的数据
    return jsonify(suggestions)




@app.route('/get_accuracy', methods=['GET'])
def get_accuracy():
    # 模拟的准确度数据，实际可以从模型或数据库中获取
    data = {
        "vector_accuracy": 75,
        "graph_accuracy": 80,
        "hybrid_accuracy": 85
    }
    return jsonify(data)


@app.route('/api/analysis_data', methods=['GET'])  # New API endpoint for analysis data
def get_analysis_data():
    # Get accuracy data from simulate.py functions
    graph_gen_faithfulness, graph_gen_accuracy = simulate.statistic_graph_generation(simulate.rgb_graph_generation)
    graph_ret_recall, graph_ret_relevance = simulate.statistic_graph_retrieval(simulate.rgb_graph_retrieval)
    vector_gen_precision, vector_gen_faithfulness, vector_gen_accuracy = simulate.statistic_vector_generation(simulate.rgb_vector_generation)
    vector_ret_precision, vector_ret_relevance, vector_ret_recall = simulate.statistic_vector_retrieval(simulate.rgb_vector_retrieval)

    # Simulate Error Statistics data (replace with your actual error stats logic later)
    error_stats_vectorrag = {'None Result': 37.7, 'Lack Information': 14.2, 'Noisy': 7.1, 'Other': 41.0}
    error_stats_graphrag = {'None Result': 69.4, 'Lack Information': 8.3, 'Noisy': 5.6, 'Other': 16.7}
    error_stats_hybridrag = {'None Result': 36.8, 'Lack Information': 9.1, 'Noisy': 9.1, 'Other': 45.0}

    # Simulate Evaluation Metrics data (replace with your actual eval metrics logic later)
    eval_metrics_vectorrag = {'precision': vector_ret_precision, 'relevance': vector_ret_relevance, 'recall': vector_ret_recall, 'faithfulness': vector_gen_faithfulness, 'accuracy': vector_gen_accuracy}
    eval_metrics_graphrag = {'precision': graph_ret_relevance, 'relevance': graph_ret_relevance, 'recall': graph_ret_recall, 'faithfulness': graph_gen_faithfulness, 'accuracy': graph_gen_accuracy}
    eval_metrics_hybridrag = {'precision': 0.6, 'relevance': 0.7, 'recall': 0.5, 'faithfulness': 0.7, 'accuracy': 0.5} # HybridRAG data needs to be filled with your logic

    analysis_data = {
        "accuracy": {
            "graphrag": round(graph_gen_accuracy * 100, 1), # Convert to percentage and round
            "vectorrag": round(vector_gen_accuracy * 100, 1),
            "hybridrag": 80 # Placeholder for HybridRAG Accuracy - replace with actual data
        },
        "errorStatistics": {
            "vectorrag": error_stats_vectorrag,
            "graphrag": error_stats_graphrag,
            "hybridrag": error_stats_hybridrag
        },
        "evaluationMetrics": {
            "vectorrag": eval_metrics_vectorrag,
            "graphrag": eval_metrics_graphrag,
            "hybridrag": eval_metrics_hybridrag
        }
    }
    print(analysis_data)
    return jsonify(analysis_data)


@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.get_json()  # 获取前端传来的 JSON 数据
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

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.get_json()  # 获取前端传来的 JSON 数据
    username = data.get('username')
    password = data.get('password')

    if not all([username, password]):
        return jsonify({"error": "用户名和密码是必需的！"}), 400

    try:
        user = User(username=username, email=None, phone=None, password=password)
        if user.login(password):
            session['username'] = username  # 将用户名存储到 session
            session['user_id'] = user.get_user_id()  # 存储用户 ID
            return jsonify({"message": "登录成功！"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400



# def web_chat(self, message : str, history: List[Optional[List]] | None, op0 : str,op1 : str, op2 : str, op3 : str):
        
#         """Used to chat with LLM from ./webchat.py

#         Args:
#             message (list[str]): The query list
#             op0 (str): The llm name ("qwen:0.5b", "llama2:7b", "llama2:13b", "llama2:70b","qwen:7b","qwen:14b","qwen:72b", etc)
#             op1 (str): The RAG way ("graph rag", "vector rag", "without rag", "union rag")
#             op2 (str): The name of graph database ("nebulagraph", "neo4j", ect)
#             op3 (str): The name of vector database ("MilvusDB")
#         """
# <div class="chat-type-selector">
#     <select id="chatType">
            # <option value="general">VectorRAG</option>
            # <option value="technical">GraphRAG</option>
            # <option value="union">Hybrid RAG</option>
            # <option value="fun">Without RAG</option>
#     </select>
#这个函数改成前端post的数据要加上 mode [Without_RAG,GraphRAG,VectorRAG]  model：str 

@app.route('/api/chat', methods=['POST'])
def web_chat():
    chat_env = LLMRAGEnv()
    data = request.get_json()
    user_input = data.get('user_input')
    model = "qwen:0.5b" #这里写死了，你们改一下
    rag_mode = data.get('mode')
    graph_db = "nebulagraph"
    vector_db = "milvus"
    history = [] 

    # 使用生成器进行流式响应
    def generate_chat():
        try:
            # 调用生成器方法，模拟生成流式数据
            answer = chat_env.web_chat(
                message=user_input,
                history=history,
                op0=model,
                op1=rag_mode,
                op2=graph_db,
                op3=vector_db
            )
            for msg in answer:
                # 按 SSE 格式发送 JSON 消息 
                print(f"data: {json.dumps({'message': msg})}")
                yield f"data: {json.dumps({'message': msg})}\n\n"
                time.sleep(0.1)  # 模拟延时

            # 流结束时，发送特殊的结束标志
            yield "data: {\"message\": \"[END]\"}\n\n"

        except Exception as e:
            # 如果发生异常，返回错误消息
            print(f"Error occurred: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


    def test_chat():
        answer = ["hello", "welcome", "to", "china"]
        for msg in answer:
            # 返回有效的 JSON 格式的消息
            print(f"data: {json.dumps({'message': msg})}")
            yield f"data: {json.dumps({'message': msg})}\n\n".encode('utf-8')
            time.sleep(1)  # 延时模拟流式输出

        # 添加结束标志
        yield "data: {\"message\": \"[END]\"}\n\n".encode('utf-8')
    

    # 返回流式数据
    return Response(generate_chat(), content_type='text/event-stream'),200



@app.route('/display_generate', methods=['GET'])
def display_generate():
    item_id = request.args.get("item_id")
    option_value = request.args.get("option_value")

    if not item_id or not option_value:
        return jsonify({"error": "Missing parameters"}), 400

    # 这里模拟数据读取
    result = {
        "query": f"Query for item {item_id} with {option_value}",
        "answer": f"Generated answer for {option_value} on item {item_id}"
    }

    return jsonify({"result": result})




# @app.route('/api/chat', methods=['POST'])
# def memory_chat():

#     data = request.get_json()  # 获取前端传来的 JSON 数据
#     user_input = data.get('user_input')
#     model = data.get('model')
    
#     client = ZhipuAI(api_key="de2e91484b4b40ebad17347e4ab33f23.L1zrAybIgusCtTKL")
#     if model == "模型1":
#         client = ZhipuAI(api_key="de2e91484b4b40ebad17347e4ab33f23.L1zrAybIgusCtTKL")
#     # elif model == "模型2":
#     #     client = ZhipuAI(api_key="your_api_key")
#     # elif model == "模型3":
#     #     client = OpenAI(api_key="your_api_key")
#     # elif model == "模型4":
#     #     client = OpenAI(api_key="your_api_key")

#     response = client.chat.completions.create(
#             model="glm-4-plus",  # 选择合适的模型
#             messages=[
#                 {"role": "user", "content": user_input}
#             ],
#         )
#     # 获取回答内容

#     answer = response.choices[0].message.content

#     user_id = session.get('user_id')  # 从 session 中获取用户 ID
#     session_id = 1      # 假设用户在进行第1个对话

#     return jsonify({"message": answer}), 200
#     '''  
#   try:
#         user = User(username=None, email=None, phone=None, password=None)
#         if not user_id:
#             return jsonify({"error": "用户未登录！"}), 401
#         if user.message_memory(session_id, user_input):
#             return jsonify({"message": "消息已保存！"}), 200
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 400
#     '''
    
@app.route('/api/get-username', methods=['GET'])
def get_username():
    username = session.get('username')  # 从 session 获取用户名
    if username:
        return jsonify({"username": username}), 200
    else:
        return jsonify({"error": "用户未登录"}), 401
    
@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()  # 清除所有 session 数据
    return jsonify({"message": "注销成功"}), 200



@app.route('/load_model', methods=['POST'])
def load_model():
    global current_model  # 使用全局变量存储模型

    try:
        data = request.json
        model_name = data.get("model_name")
        url = data.get("url")
        key = data.get("key")
        dataset = data.get("dataset")
        if key == "":
            key = "ollama"
        print(model_name,key)

        if not model_name or not key:
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400

        # 如果当前已有模型运行，则先关闭
        if current_model is not None:
            try:
                current_model.close()  # 关闭当前模型（假设ClientFactory的客户端有close()方法）
                print(f"已关闭模型: {current_model}")
            except Exception as e:
                print(f"关闭模型失败: {e}")

        # 加载新模型
        current_model = Demo_chat(model_name=model_name,api_key=key,dataset=dataset)
        print(current_model.chat_test())

        return jsonify({"status": "success", "message": f"模型 {model_name} 加载成功"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)