import json
import os


def check_json_file(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    elif not isinstance(file_paths, list):
        raise TypeError("file_paths should be a string or a list of strings.")
    for file_path in file_paths:
        if not file_path.lower().endswith('.json'):
            raise ValueError(
                f"Error: The file '{file_path}' is not a JSON file. Please provide a valid JSON file.")


def get_question_list(question_data_dir):
    try:
        with open(question_data_dir, 'r') as f:
            load_data = json.load(f)
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return []
    except IOError:
        print("Error: An I/O error occurred while accessing the file.")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")
        return []

    question_list = []
    for instance in load_data:
        question_list.append(instance)
    print(f"question_list length:{len(question_list)}")
    return question_list


def get_result_list(result_data_dir):
    return get_question_list(result_data_dir)


def get_unique_context_list(data_path):
    all_question_info = get_question_list(data_path)
    all_context = []
    all_question_info = all_question_info
    for item in all_question_info:
        all_context.extend(item["context"].values())
    return list(set(all_context))


def get_file_directory_path(file_path):
    return os.path.dirname(file_path)


def create_json_by_path_located_directory(path_dir, file_name):
    if not isinstance(path_dir, str):
        print(type(path_dir))
        raise TypeError("f{path_dir} should be a string.")

    directory = os.path.dirname(path_dir)

    if not os.path.isdir(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")

    if os.path.isabs(file_name):  # avoid absolute path
        file_name = file_name.lstrip("/")

    create_file_path = os.path.join(directory, file_name)
    full_directory = os.path.dirname(create_file_path)
    if not os.path.isdir(full_directory):
        os.makedirs(full_directory, exist_ok=True)

    with open(create_file_path, 'w') as retry_file:
        json.dump({}, retry_file)
    print(f"'{file_name}' has been created at: {create_file_path}")
    return create_file_path


def get_filename_without_extension(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    return file_name_without_extension


def get_rgb_or_multihop_question_list(question_data_dir):
    question_list = []
    with open(question_data_dir, 'r') as f:
        for line in f:
            question_list.append(json.loads(line))
    return question_list


# def get_multihop_question_list(question_data_dir):
#     question_list = []
#     with open(question_data_dir, 'r') as f:
#         for line in f:
#             question_list.append(json.loads(line))
#     return question_list

def get_arxiv_or_report_question_list(question_data_dir):
    question_list = []
    with open(question_data_dir, 'r') as f:
        load_data = json.load(f)
    for data in load_data:
        question_list.append(data)
    return question_list


# def get_report_question_list(question_data_dir):
#     question_list = []
#     with open(question_data_dir, 'r') as f:
#         load_data = json.load(f)
#     for data in load_data:
#         question_list.append(data)
#     return question_list


def find_extract_entity_empty_question(question_data_dir):
    question_list = []
    not_empty_entity_question_list = []
    empty_entity_question_list = []
    with open(question_data_dir, 'r') as f:
        for line in f:
            question_list.append(json.loads(line))
    for question in question_list:
        entities = question['entities']
        if entities == "":
            empty_entity_question_list.append(question)
        else:
            not_empty_entity_question_list.append(question)
    print(
        f"not_empty_entity_question_list: {len(not_empty_entity_question_list)}, empty_entity_question_list: {len(empty_entity_question_list)}")
    return not_empty_entity_question_list, empty_entity_question_list


def find_lack_question(origin_question_path, processed_question_path):
    origin_question_list = get_rgb_or_multihop_question_list(
        origin_question_path)
    processed_question_list = get_rgb_or_multihop_question_list(
        processed_question_path)
    lack_question_list = []

    processed_question_dis = {item['id'] for item in processed_question_list}
    lack_question_list = [
        item for item in origin_question_list if item['id'] not in processed_question_dis]
    print(f"lack_question_list: {len(lack_question_list)}")
    return lack_question_list


def find_extract_entity_empty_question_test(question_data_dir):
    question_list = []
    try:
        with open(question_data_dir, 'r', encoding='utf-8') as f:
            for line in f:
                question = json.loads(line.strip())  # 去除行尾的换行符并解析 JSON
                question_list.append(question)
    except Exception as e:
        print(f"Error: {e}")
        return [], []


if __name__ == "__main__":
    # home_dir = os.path.expanduser("~")
    # question_data_dir = os.path.join(home_dir,
    #                                 'NeutronRAG/result/graphrag_response/arxiv_graph_llama2:70b.json')
    # # find_extract_entity_empty_question(question_data_dir)

    # multihop_original_question_data_dir = os.path.join(home_dir,
    #                                 'NeutronRAG/external_corpus/multihop_data/MultiHop_RAG_main/dataset/process.json')
    # multihop_processed_question_data_dir= os.path.join(home_dir,
    #                                 'NeutronRAG/result/graphrag_response/multihop_graph_llama2:70b.json')
    # question_data_dir = '/home/chency/NeutronRAG/result/hybridrag_response/rgb_vector_graph_llama2:70b.json'
    # find_lack_question(multihop_original_question_data_dir, multihop_processed_question_data_dir)
    # find_extract_entity_empty_question_test(question_data_dir)
    result_store_current = "/home/chency/NeutronRAG/neutronrag/results/rgb/generation"
    # create_json_by_path_located_directory(result_store_current, "./test.json")

    data = "rgb"
    engine_name = "graph"
    graph_k_hop = 10
    graph_retrieval_limit = 10
    result_store_path = os.path.join(
        "/home/chency/NeutronRAG/neutronrag/results/generation", f"{data}/{engine_name}rag_response/")
    result_store_path = os.path.join(
        result_store_path, f"khop:{graph_k_hop}_pruning:{graph_retrieval_limit}.json")
    print(result_store_path)
