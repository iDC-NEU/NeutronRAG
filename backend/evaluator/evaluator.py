'''
Author: lpz 1565561624@qq.com
Date: 2025-03-25 22:46:59
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-27 09:50:19
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/evaluator/evaluator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEn
'''
from generation_metric import *
from retrieval_metric import *
from ragas_evalutor import *
from file_operation import *
from get_question_list import *
from dbname_map import get_data_type

home_dir = os.path.expanduser("~")
multihop_data_path = os.path.join(
    home_dir, "NeutronRAG/external_corpus/multihop_data/MultiHop_RAG_main/dataset/corpus.json")


def get_mutihop_by_key_value():
    # key is url, value is body
    multihop_data = {}
    with open(multihop_data_path, 'r') as file:
        load_data = json.load(file)
    for data in load_data:
        multihop_data[data["url"]] = data["body"]
    print(f" multihop document length: {len(multihop_data)}")
    return multihop_data


def get_id_map_quetion_list(question_data_dir):
    question_list = {}
    with open(question_data_dir, 'r') as f:
        load_data = json.load(f)
    for data in load_data:
        if data['id'] not in question_list:
            question_list[data['id']] = data
        else:
            print(f" question id:{data['id']}")
    return question_list


class Evaluator:
    def __init__(self,
                 data_name,
                 mode):
        self.data_name = data_name
        self.data_type = get_data_type(self.data_name)

        if self.data_name == "multihop":
            self.url_body_multihop = get_mutihop_by_key_value()

        self.mode = mode
        self.qa_evalutor = QAGenerationEvalutor()
        self.summary_evalutor = SummaryGenerationEvalutor()
        self.ragas_evalutor = RAGASEvalutor()
        self.retrieval_evalutor = RetrievalEvaluator()

    def evaluate_at_specified_stage(self, evaluation_list, metric, analysis_file_path):
        self.data_list = evaluation_list
        print(type(self.data_list))
        print(len(evaluation_list))
        if metric == "generation":
            self.generation_metric(analysis_file_path)
        elif metric == "retrieval":
            self.evaluate_retrieval_with_precision_recall(
                analysis_file_path)
        elif metric == "both":
            self.generation_metric(analysis_file_path)
            self.evaluate_retrieval_with_precision_recall(analysis_file_path)

        # elif metric == "compare_vector_graph":
            # evaluator.compare_rag_by_exact_match(vectorrag_result_path, graphrag_result_path, vectorrag_better_path, graphrag_better_path)
        # elif metric == "compare_all":
        #     # evaluator.compare_rag_by_exact_match(vectorrag_result_path, graphrag_result_path, vectorrag_better_path, graphrag_better_path, hybridrag_result_path, hybridrag_better_path)
        #     pass
        else:
            raise ValueError(
                "Error: metric parameter is not valid. It must be either 'generation' or'retrieval'.")

    def evaluate_retrieval_with_precision_recall(self, store_path, is_open_ragas=False, store_all_path=None):
        eval_reuslt_list = []
        question_list = []
        precision_scores = 0
        recall_scores = 0
        relevance_scores = 0
        for question in self.data_list:
            eval_question = {}
            eval_question['id'] = question['id']

            question_str = self.get_question_str(question)
            ground_truth = self.get_response_str(question)

            eval_question["question"] = question_str
            eval_question["answer"] = ground_truth
            eval_question["response"] = question["response"]
            eval_question["evidences"] = question["evidences"]
            eval_question["retrieve_results"] = question["retrieve_results"]

            ground_truth_context = self.get_target_retrieval_texts(question)
            if isinstance(ground_truth_context, str):
                ground_truth_context = [ground_truth_context]
            retrieved_context = self.get_retrieval_context(question)

            eval_question["retrieval_evaluation"] = {}
            eval_question["retrieval_evaluation"]["precision"] = self.retrieval_evalutor.evaluation_precision(
                retrieved_context, ground_truth_context)
            eval_question["retrieval_evaluation"]["relevance"] = self.retrieval_evalutor.evaluation_relevance(
                retrieved_context, ground_truth_context, mode=self.mode)
            eval_question["retrieval_evaluation"]["recall"] = self.retrieval_evalutor.evaluation_recall(
                retrieved_context, ground_truth_context, mode=self.mode)
            question_list.append(question)
            eval_reuslt_list.append(eval_question)
            precision_scores += eval_question["retrieval_evaluation"]["precision"]
            recall_scores += eval_question["retrieval_evaluation"]["recall"]
            relevance_scores += eval_question["retrieval_evaluation"]["relevance"]
        save_response(eval_reuslt_list, store_path)
        print(
            f"precision avg: {precision_scores/len(eval_reuslt_list)}, recall avg: {recall_scores/len(eval_reuslt_list)}, relevance avg: {relevance_scores/len(eval_reuslt_list)}")
        if store_all_path:
            save_response(question_list, store_all_path)

    # retrieval evaluation by ragas
    def retrieval_metric(self, store_path, is_open_ragas=False, store_all_path=None):
        only_eval_reuslt_list = []
        question_list = []
        for question in self.data_list:
            eval_question = {}
            eval_question['id'] = question['id']

            question_str = self.get_question_str(question)
            ground_truth = self.get_response_str(question)
            predict_output = question["response"]

            target_retrieval_texts = self.get_target_retrieval_texts(question)
            if isinstance(target_retrieval_texts, str):
                target_retrieval_texts = [target_retrieval_texts]
            retrieval_context = self.get_retrieval_context(question)

            eval_question["question"] = question_str
            eval_question["answer"] = ground_truth

            # if self.mode == "vector":
            #     eval_question["retrieval_precision"] = self.retrieval_evalutor.precision_rate_vector_rag(retrieval_context, target_retrieval_texts)
            # elif self.mode == "graph":
            #     eval_question["retrieval_precision"] = self.retrieval_evalutor.precision_rate_graph_rag(retrieval_context, target_retrieval_texts)
            # # elif self.mode == "vector_graph":
            #     eval_question["retrieval_precision"] = self.retrieval_evalutor.precision_rate_vector_rag(retrieval_context, target_retrieval_texts)

            # if is_open_ragas:
            process_question_str, process_ground_truth, process_predict_output, process_retrieval_context = self.ragas_preprocess(
                question)
            ragas_score = self.ragas_evalutor.evaluate_retrieval_single_question(
                process_question_str, process_ground_truth, process_predict_output, process_retrieval_context)
            question["ragas_retrieval_score"] = ragas_score
            eval_question["ragas_retrieval_score"] = ragas_score

            question_list.append(question)
            only_eval_reuslt_list.append(eval_question)
        save_response(only_eval_reuslt_list, store_path)
        if store_all_path:
            save_response(question_list, store_all_path)

    def generation_metric(self, store_path, is_open_ragas=True, store_all_path=None):
        eval_reuslt_list = []
        question_list = []
        hallucinations_total_score = 0
        hallucinations_count = 0
        hallucinations_error_list = []
        for question in self.data_list:
            eval_question = {}
            eval_question['id'] = question['id']

            question_str = self.get_question_str(question)
            ground_truth = self.get_response_str(question)
            predict_output = question["response"]
            process_question_str, process_ground_truth, process_predict_output, process_retrieval_context = self.ragas_preprocess(
                question)

            eval_question["question"] = question_str
            eval_question["answer"] = ground_truth
            eval_question["response"] = predict_output

            question["generation_evaluation"] = {}
            eval_question["generation_evaluation"] = {}

            if self.data_type == 'qa':
                exact_match = self.qa_evalutor.checkanswer(
                    predict_output, ground_truth)
                question["generation_evaluation"]["exact_match"] = exact_match
                eval_question["generation_evaluation"]["exact_match"] = exact_match
            elif self.data_type == 'summary':
                rouge_score = self.summary_evalutor.get_rougeL_score(
                    ground_truth, predict_output)
                question["generation_evaluation"]["rougeL_score"] = rouge_score
                eval_question["generation_evaluation"]["rougeL_score"] = rouge_score
                anwser_accuracy_by_ragas = self.ragas_evalutor.evaluate_answer_accuracy(
                    process_ground_truth, process_predict_output)
                question["generation_evaluation"]["answer_correctness"] = anwser_accuracy_by_ragas
                eval_question["generation_evaluation"]["answer_correctness"] = anwser_accuracy_by_ragas

            # evaluate hallucinations
            hallucinations_score = -1
            if is_open_ragas:
                hallucinations_score = self.ragas_evalutor.evaluate_hallucinations(
                    process_question_str, process_predict_output,  process_retrieval_context)
                # hallucinations_score = self.ragas_evalutor.evaluate_generation_single_question(
                #     process_question_str, process_ground_truth, process_predict_output, process_retrieval_context, eval_metrics=[faithfulness])
                question["generation_evaluation"]["hallucinations"] = hallucinations_score
                eval_question["generation_evaluation"]["hallucinations"] = hallucinations_score
                if hallucinations_score != -1:
                    hallucinations_total_score += hallucinations_score
                    hallucinations_count += 1
                else:
                    hallucinations_error_list.append(eval_question)
                print(f"question : {process_question_str}")
                print(f"hallucnations_score:{hallucinations_score}\n")

            question_list.append(question)
            eval_reuslt_list.append(eval_question)
        avg_score = 0
        if is_open_ragas:
            avg_score = hallucinations_total_score/hallucinations_count
        print(
            f"total_score:{hallucinations_total_score}, available_count:{hallucinations_count}, avg_hallucinations_score:{(avg_score)}\n")
        save_response(hallucinations_error_list, create_json_by_path_located_directory(
            store_path, f"/hallucinations_error.json"))
        save_response(eval_reuslt_list, store_path)
        self.count_right_length(eval_reuslt_list)

        if store_all_path:
            save_response(question_list, store_all_path)

    def count_right_length(self, only_eval_reuslt_list):
        right_length = 0
        if self.data_type == 'qa':
            for item in only_eval_reuslt_list:
                label = item["generation_evaluation"]['exact_match']
                if label > 0:
                    right_length += 1
        elif self.data_type == 'summary':
            for item in only_eval_reuslt_list:
                rouge_score = item["generation_evaluation"]['rougeL_score']
                if rouge_score >= 0.5:
                    right_length += 1
        else:
            raise ValueError(
                "Error: data type is not valid. It must be either 'qa' or'summary'.")
        print(
            f"dataset: {self.data_name}, total_length: {len(only_eval_reuslt_list)}, right_length: {right_length}")

    def compare_rag_by_exact_match(self, vectorrag_result_path, graphrag_result_path, vectorrag_better_path, graphrag_better_path, hybridrag_result_path=None, hybridrag_better_path=None):
        vector_question_list = get_id_map_quetion_list(vectorrag_result_path)
        graph_question_list = get_id_map_quetion_list(graphrag_result_path)

        # keys_600 = set(vector_question_list.keys())
        # keys_599 = set(graph_question_list.keys())

        # # Find the difference between the sets
        # missing_key = keys_600 - keys_599

        # # Return the missing key (there should only be one)
        # if missing_key:
        #     print(f"The missing ID is: {missing_key}")
        # else:
        #     return None  # No missing key found

        vectorrag_better_list = []
        graphrag_better_list = []

        if hybridrag_result_path:
            hybridrag_question_list = get_id_map_quetion_list(
                hybridrag_result_path)
            hybridrag_better_list = []
            assert len(vector_question_list) == len(
                graph_question_list) == len(hybridrag_question_list)
        else:
            print(len(vector_question_list), len(graph_question_list))
            assert len(vector_question_list) == len(graph_question_list)

        graph_score = 0
        vector_score = 0
        for key, value in vector_question_list.items():
            if self.data_type == 'qa':
                vector_label = value['lable']
                graph_label = graph_question_list[key]['lable']
                vector_right_count = vector_label.count(1)
                graph_right_count = graph_label.count(1)
            elif self.data_type == 'summary':
                vector_right_count = value['ragas_generation_score']['answer_relevancy']
                vector_score += vector_right_count
                graph_right_count = graph_question_list[key]['ragas_generation_score']['answer_relevancy']
                graph_score += graph_right_count

            if hybridrag_result_path:
                if self.data_type == 'qa':
                    hybrid_label = hybridrag_question_list[key]['lable']
                    hybrid_right_count = hybrid_label.count(1)
                elif self.data_type == 'summary':
                    hybrid_right_count = hybridrag_question_list[key][
                        'ragas_generation_score']['answer_relevancy']

                if vector_right_count > graph_right_count and vector_right_count > hybrid_right_count:
                    vectorrag_better_list.append(value)
                elif graph_right_count > vector_right_count and graph_right_count > hybrid_right_count:
                    graphrag_better_list.append(graph_question_list[key])
                elif hybrid_right_count > vector_right_count and hybrid_right_count > graph_right_count:
                    hybridrag_better_list.append(hybridrag_question_list[key])
            else:
                if vector_right_count > graph_right_count:
                    vectorrag_better_list.append(value)
                elif graph_right_count > vector_right_count:
                    graphrag_better_list.append(graph_question_list[key])
                elif graph_right_count == vector_right_count:
                    print("================================")
        print(
            f"vector rag right len:{len(vectorrag_better_list)}, graph rag right len:{len(graphrag_better_list)}")
        print(f"vector_score:{vector_score}, graph_score:{graph_score}")
        if hybridrag_result_path:
            print(f", hybrid rag right len:{len(hybridrag_better_list)}")

        save_response(vectorrag_better_list, vectorrag_better_path)
        save_response(graphrag_better_list, graphrag_better_path)
        if hybridrag_result_path:
            save_response(hybridrag_better_list, hybridrag_better_path)

    def ragas_preprocess(self, question):
        question_str = self.get_question_str(question)
        ground_truth = self.get_response_str(question)
        predict_output = question["response"]
        retrieval_context = self.get_retrieval_context(question)

        if self.data_name == "rgb":
            ground_truth = self.list_to_string(ground_truth)
        # print(f"ground_truth:{ground_truth}")
        # print(f"retrieval_context:{retrieval_context}")

        return question_str, ground_truth, predict_output, retrieval_context

    def list_to_string(self, data):
        if all(isinstance(i, list) for i in data):
            # 如果所有元素都是列表，处理嵌套列表
            sub_lists = [" or ".join(sublist) for sublist in data]
            result = ", ".join(sub_lists)
        else:
            # 处理普通列表
            result = ", ".join(data)
        return result

    def get_question_str(self, data):
        if 'query' in data:
            return data['query']
        elif 'question' in data:
            return data['question']
        else:
            raise ValueError(
                "Neither 'query' nor 'question' key found in dictionary")

    def get_response_str(self, data):
        if 'answer' in data:
            return data['answer']
        elif 'summary' in data:
            return data['summary']
        else:
            raise ValueError(
                "Neither 'answer' nor 'summary' key found in dictionary")

    def get_retrieval_context(self, data):
        if self.mode == "graph":
            process_retrieval_texts = [
                path_text for each_graph in data['retrieve_results'].values() for path_text in each_graph]

            print("-##########", process_retrieval_texts)
            return process_retrieval_texts
        elif self.mode == "vector":
            process_retrieval_texts = [item['node_text']
                                       for item in data['retrieve_results']]
            return process_retrieval_texts
        elif self.mode == "vector_graph":
            graph_retrieval_texts = data['graph_retrieve_results']['pruning']
            print(f"graph_retrieval_texts:{graph_retrieval_texts}")
            vector_retrieval_texts = data['vector_retrieve_results']
            process_vector_retrieval_texts = [
                item['node_text'] for item in vector_retrieval_texts]
            print(
                f"process_vector_retrieval_texts:{process_vector_retrieval_texts}")
            return graph_retrieval_texts + process_vector_retrieval_texts
        else:
            raise ValueError(
                "Neither 'answer' nor 'summary' key found in dictionary")

    def get_target_retrieval_texts(self, data):
        if self.mode == "vector":
            if isinstance(data['evidences'], str):
                return [data['evidences']]
            return data['evidences'].values()
        elif self.mode == "graph":
            if "evidences_triplets" in data:
                triplets = data['evidences_triplets']
                flat_triplets = list(
                    set([item for sublist in triplets for item in sublist]))
                return flat_triplets

        elif self.mode == "vector_graph":
            pass
        else:
            raise NameError("f{self.mode} doset not exit")
        # if 'evidences' in data:
        #     return data['evidences']
        # if 'source' in data:
        #     return data['source']
        # if 'positive' in data:
        #     return data['positive']
        # if 'evidence_list' in data:
        #     target_text = [evidence["url"]
        #                    for evidence in data['evidence_list']]
            return target_text



#    {
#         "id": 284,
#         "query": "Which city will host The World Games 2025?",
#         "answer": [
#             "Chengdu"
#         ],
#         "is_true": true,
#         "merged_triplets": [
#             [
#                 [
#                     "Chengdu",
#                     "Will_be",
#                     "The host city for the world games 2025"
#                 ],
#                 [
#                     "The world games 2025",
#                     "Will_take_place_in",
#                     "Chengdu"
#                 ],
#                 "The host city for the world games 2025 <-Will_be- Chengdu",
#                 "Chengdu <-Will_take_place_in- The world games 2025"
#             ]
#         ]
#     }

#{
#        "id": 83,
#        "query": "Who won the 2022 Tour de France?",
#        "answer": [
#            "Jonas Vingegaard"
#        ],
#        "evidences": "Your stage-by-stage guide to the winners of the 2022 Tour. Denmark's Jonas Vingegaard (Jumbo-Visma) won the yellow jersey as the overall winner of the 2022 Tour de France. The 25-year-old outlasted two-time defending champion Tadej Pogačar (UAE Team Emirates) of Slovenia to win his first Tour. Pogačar finished second, 2:43 back of Vingegaard, and Great Britain's Geraint Thomas (INEOS Grenadiers) was third, 7:22 behind the lead, to round out the podium for the Tour's General Classification. Here’s a look at how every stage of the 2022 Tour unfolded.  Results From Every Stage Full Leaderboard Who Won the Tour? Surrounded by his teammates, Denmark’s Jonas Vingegaard (Jumbo-Visma) finished safely behind the peloton at the end of Stage 21 in Paris to win the 2022 Tour de France. The Dane won the Tour by 3:34 over Slovenia’s Tadej Pogačar (UAE Team Emirates), who started the race as the two-time defending champion, and 8:13 over Great Britain’s Geraint Thomas (INEOS Grenadiers), who won the Tour in 2018 and finished second in 2019."







#先以RGB为例
# rgb_evidence_vector = ""
#rgb_evidence_graph = ""
    def evaluate_one_query(self,query_id:int,query:str,response:str,retrieval_result:list,vector_evidence,graph_evidence):
        # precision_scores = 0
        # recall_scores = 0
        # relevance_scores = 0
        # answer_correctness = 0
        # rougeL_score = 0

        try:
            with open(vector_evidence, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)
            with open(graph_evidence, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error loading evidence files: {e}") 
            return None
        
        vector_evidence_item = [item for item in vector_data if item.get('id') == query_id][0]
        graph_evidence_item = [item for item in graph_data if item.get('id') == query_id][0]
        ground_truth = vector_evidence_item["answer"]
        if self.data_name == "rgb":
            ground_truth = self.list_to_string(ground_truth)

        if not vector_evidence_item and not graph_evidence_item:
            print(f"No evidence found for query ID {query_id}")
            return None
        
        print("is checkanswer")
        if self.data_type == 'qa':
            exact_match = self.qa_evalutor.checkanswer(
                    response, ground_truth)
        elif self.data_type == 'summary':
            rouge_score = self.summary_evalutor.get_rougeL_score(
                    ground_truth, response)
            anwser_accuracy_by_ragas = self.ragas_evalutor.evaluate_answer_accuracy(
                    ground_truth, response)
            rougeL_score = rouge_score
            answer_correctness = anwser_accuracy_by_ragas
        
        print("is evaluate_hallucinations")
        hallucinations_score = -1
        
        # hallucinations_score = self.ragas_evalutor.evaluate_hallucinations(
        #             query, response,  retrieval_result)
        
        ground_truth_context = vector_evidence_item["evidences"]
        if isinstance(ground_truth_context, str):
            ground_truth_context = [ground_truth_context]
        else:
            ground_truth_context = list(ground_truth_context.values())
        ground_truth_context = set(ground_truth_context)

        print("is evaluation_precision")
        precision_scores = self.retrieval_evalutor.evaluation_precision(
                retrieval_result, ground_truth_context)
        relevance_scores =   self.retrieval_evalutor.evaluation_relevance(
                retrieval_result, ground_truth_context, mode=self.mode)
        recall_scores = self.retrieval_evalutor.evaluation_recall(
                retrieval_result, ground_truth_context, mode=self.mode)
        
        result = {
        "strategy":"vector",
        "metrics": {
            "retrieval_metrics": {
                "precision": {precision_scores},
                "recall": {recall_scores},
                "relevance": {relevance_scores}
            },
            "generation_metrics": {
                "answer_correctness": 0,
                "rougeL_score": 0,
                "hallucinations_score": {hallucinations_score},
                "exact_match": {exact_match}
            }
        }
    }
        
        return result
        
        




        
            
            

# def list_to_string(data):
#     if all(isinstance(i, list) for i in data):
#         # 如果所有元素都是列表，处理嵌套列表
#         sub_lists = [" or ".join(sublist) for sublist in data]
#         result = ", ".join(sub_lists)
#     else:
#         # 处理普通列表
#         result = ", ".join(data)
#     return result

# # 示例数据
# nested_list = [["May 11", "5 11"], ["22", "twenty two"]]
# simple_list = ["May 11", "5 11", "22", "twenty two"]

# # 处理嵌套列表
# print(list_to_string(nested_list))  # 输出：May 11 or 5 11, 22 or twenty two

# # 处理普通列表
# print(list_to_string(simple_list))  # 输出：May 11, 5 11, 22, twenty two

if __name__ == '__main__':

    data = {"retrieve_results": {
        "Admission records": [
            "Admission records <-Signed- Dr. j. edwards -Signed-> Admission_records",
            "Admission records <-Signed- Dr. j. edwards"
        ],
        "O. myers": [
            "O. myers -Is associated with-> Clarksville general hospital",
            "O. myers -Has a general health condition of-> Good health prior to current illness",
            "O. myers -Was admitted with complaints of-> Palpitations and excessive sweating <-Presents with- O. myers",
            "O. myers -Was admitted with complaints of-> Palpitations and excessive sweating",
            "O. myers -Has symptoms that are-> Persistent and worsening",
            "O. myers -Lives in-> Clarksville",
            "O. myers -Was previously seen by-> A primary care physician",
            "O. myers -Presents with-> Palpitations and excessive sweating <-Was admitted with complaints of- O. myers",
            "O. myers -Was born in-> Clarksville",
            "O. myers -Lives in-> Clarksville <-Was born in- O. myers",
            "O. myers -Has a known disease history of-> Hyperthyroidism for 2 weeks",
            "O. myers -Was born in-> Clarksville <-Lives in- O. myers",
            "O. myers -Is monitored every-> 3 days <-Had high fever occurring for- L. brown",
            "O. myers -Is-> A female patient",
            "O. myers -Is-> A female patient <-Is- H. nelson",
            "O. myers -Has a diagnostic basis including-> History of hyperthyroidism, physical examination, and laboratory tests",
            "O. myers -Has a diagnosis of-> Hyperthyroidism",
            "O. myers -Is-> A female patient <-Is- S. moore",
            "O. myers -Reports accompanying symptoms including-> Tachycardia, increased appetite, fatigue, goiter, and exophthalmos",
            "O. myers -Reports main symptoms including-> Palpitations, excessive sweating, and irritability",
            "O. myers -Resides at-> 39 woodland street, clarksville",
            "O. myers -Was previously seen by-> A primary care physician <-Was evaluated by- L. bailey",
            "O. myers -Was previously seen by-> A primary care physician <-Previously consulted- J. reyes",
            "O. myers -Is monitored every-> 3 days <-Has had persistent main symptoms for- Q. gomez",
            "O. myers -Is-> A female patient <-Is- E. parker",
            "O. myers -Has symptoms with-> Gradual onset",
            "O. myers -Has experienced palpitations and excessive sweating for a duration of-> 2 weeks <-Has duration of- Ear discharge",
            "O. myers -Has experienced palpitations and excessive sweating for a duration of-> 2 weeks",
            "O. myers -Started experiencing symptoms-> 2 weeks ago at home",
            "O. myers -Started experiencing symptoms-> 2 weeks ago at home <-Showed onset of symptoms- S. moore"
        ],
        "Clarksville general hospital": [
            "Clarksville general hospital <-Is associated with- O. myers -Was recorded on-> 20th april",
            "Clarksville general hospital <-Is associated with- O. myers -Was admitted on-> 20th april",
            "Clarksville general hospital <-Is associated with- O. myers",
            "Clarksville general hospital <-Is associated with- O. myers -Has a general health condition of-> Good health prior to current illness",
            "Clarksville general hospital <-Is associated with- O. myers -Lives in-> Clarksville",
            "Clarksville general hospital <-Is associated with- O. myers -Was born in-> Clarksville",
            "Clarksville general hospital <-Is associated with- O. myers -Has symptoms that are-> Persistent and worsening",
            "Clarksville general hospital <-Is associated with- O. myers -Is stable in-> The current condition",
            "Clarksville general hospital <-Is associated with- O. myers -Was admitted with complaints of-> Palpitations and excessive sweating",
            "Clarksville general hospital <-Is associated with- O. myers -Was previously seen by-> A primary care physician",
            "Clarksville general hospital <-Is associated with- O. myers -Resides at-> 39 woodland street, clarksville",
            "Clarksville general hospital <-Is associated with- O. myers -Started experiencing symptoms-> 2 weeks ago at home",
            "Clarksville general hospital <-Is associated with- O. myers -Is monitored every-> 3 days",
            "Clarksville general hospital <-Is associated with- O. myers -Has symptoms with-> Gradual onset",
            "Clarksville general hospital <-Is associated with- O. myers -Reports accompanying symptoms including-> Tachycardia, increased appetite, fatigue, goiter, and exophthalmos",
            "Clarksville general hospital <-Is associated with- O. myers -Reports main symptoms including-> Palpitations, excessive sweating, and irritability",
            "Clarksville general hospital <-Is associated with- O. myers -Has healthy-> Living habits",
            "Clarksville general hospital <-Is associated with- O. myers -Has a diagnostic basis including-> History of hyperthyroidism, physical examination, and laboratory tests",
            "Clarksville general hospital <-Is associated with- O. myers -Is-> 13 years old",
            "Clarksville general hospital <-Is associated with- O. myers -Is-> A female patient",
            "Clarksville general hospital <-Is associated with- O. myers -Has a pulse rate of-> 110 bpm",
            "Clarksville general hospital <-Is associated with- O. myers -Has a respiration rate of-> 20 breaths/min",
            "Clarksville general hospital <-Is associated with- O. myers -Has a current treatment plan to continue with-> Close monitoring of thyroid function",
            "Clarksville general hospital <-Is associated with- O. myers -Has a diagnosis of-> Hyperthyroidism",
            "Clarksville general hospital <-Is associated with- O. myers -Is-> Single",
            "Clarksville general hospital <-Is associated with- O. myers -Had an endocrinology consultation requested by-> The management plan",
            "Clarksville general hospital <-Is associated with- O. myers -Has a known disease history of-> Hyperthyroidism for 2 weeks",
            "Clarksville general hospital <-Is associated with- O. myers -Has blood pressure recorded as-> 110/70 mmhg",
            "Clarksville general hospital <-Is associated with- O. myers -Had her last menstruation on-> 15th april",
            "Clarksville general hospital <-Is associated with- O. myers -Had senior physician rounds emphasizing-> The need for close monitoring"
        ],
        "The admission record": [
            "The admission record <-Signed- Dr. emily watson"
        ]
    }}

    process_retrieval_texts = [
        path_text for each_graph in data['retrieve_results'].values() for path_text in each_graph]

    # print(process_retrieval_texts)

    evalutor_test = Evaluator(data_name="rgb",mode="vector")
#  /home/chency/NeutronRAG/neutronrag/results/response/rgb/vectorrag/top5_2024-11-26_21-32-23.json
    retrieve_results = ["Your stage-by-stage guide to the winners of the 2022 Tour","Your stage-by-stage guide to the winners of the 2022 Tour. Denmark's Jonas Vingegaard (Jumbo-Visma) won the yellow jersey as the overall winner of the 2022 Tour de France. The 25-year-old outlasted two-time defending champion Tadej Pogačar (UAE Team Emirates) of Slovenia to win his first Tour. Pogačar finished second, 2:43 back of Vingegaard, and Great Britain's Geraint Thomas (INEOS Grenadiers) was third, 7:22 behind the lead, to round out the podium for the Tour's General Classification. Here’s a look at how every stage of the 2022 Tour unfolded.  Results From Every Stage Full Leaderboard Who Won the Tour? Surrounded by his teammates, Denmark’s Jonas Vingegaard (Jumbo-Visma) finished safely behind the peloton at the end of Stage 21 in Paris to win the 2022 Tour de France. The Dane won the Tour by 3:34 over Slovenia’s Tadej Pogačar (UAE Team Emirates), who started the race as the two-time defending champion, and 8:13 over Great Britain’s Geraint Thomas (INEOS Grenadiers), who won the Tour in 2018 and finished second in 2019."]
    result = evalutor_test.evaluate_one_query(query_id=83,query="Who won the 2022 Tour de France?",retrieval_result=retrieve_results,response="Denmark's Jonas Vingegaard (Jumbo-Visma) won the yellow jersey as the overall winner of the 2022 Tour de France.",
                                     vector_evidence="rgb_evidence_test.json",graph_evidence="rgb_evidence_test.json")
    

    print(result)
    

