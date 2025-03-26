import os
import statistics
from collections import defaultdict
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import CorrectnessEvaluator
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import *
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from typing import Literal
import asyncio
import math
import warnings
import openai
import json
import re


'''
    ragas metric:  9 个指标
        __all__ = [
            "answer_correctness",
            "faithfulness",
            "answer_similarity",
            "context_precision",
            "context_utilization",
            "context_recall",
            "AspectCritique",
            "context_relevancy",
            "answer_relevancy",
            "context_entity_recall",
        ]

    document: https://docs.ragas.io/en/stable/concepts/metrics/index.html
'''


'''
    指标划分：
        生成指标：
        检索指标：




'''

# os.environ["OPENAI_API_KEY"] = "sk-oVZN2wP5QXKDEctG0eF605372b624dC9812eEf02Cc9eE04e"
# os.environ["OPENAI_API_BASE"] = "https://gitaigc.com/v1"
# model = "gpt-3.5-turbo"

os.environ["OPENAI_API_KEY"] = "sk-6jOxYQjNjn9E1FMb9KlGo18gAWrWqfetKnUzQ4O6F5d7t23P"
os.environ["OPENAI_API_BASE"] = "https://api.aigc798.com/v1/"
model = "gpt-4o-mini"


prompt_expand_answer_str = """
### Task: Expand Answer into a Complete Sentence

#### Objective:
Your task is to rewrite the provided answer into a clear, complete, and grammatically correct sentence. Ensure the output sentence is self-contained and understandable without referencing the original question. However, you must strictly adhere to the original answer without altering its content or generating new information based on external knowledge. 

#### Response Format:
Your final response should follow this format:

```json
{{
    "expanded_sentence": "The expanded complete sentence."
}}
```

#### Example:

**Input Question:**
Who is the president of the Prospect Park Alliance?

**Input Answer:**
Morgan Monaco.

**Output:**
```json
{{
    "expanded_sentence":"The president of the Prospect Park Alliance is Morgan Monaco."
}}
```

#### Instructions:
**YYour task is to rewrite the answer into a complete sentence.

**Input Question:**
{input_question}

**Input Answer:**
{input_answer}

**Output:**

"""


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(),
                      re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()


class OpenAIExpander:
    def __init__(self):
        # initial OpenAI configuration
        openai.api_key = os.environ["OPENAI_API_KEY"]
        openai.base_url = os.environ["OPENAI_API_BASE"]
        self.model = model
        self.client = openai

    def expand_response_to_sentence(self, question_str, response_str, temperature=0) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_expand_answer_str.format(
                    input_question=question_str, input_answer=response_str)}],
                temperature=temperature
            )
            response = response.choices[0].message.content
            return self.get_pasrse_output(response.strip(), field="expanded_sentence")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def get_pasrse_output(self, output_str, field=Literal["expanded_sentence"]):
        retry = 3
        while retry > 0:
            try:
                output_data = json.loads(extract_json_str(output_str))
                assert field in output_data
                if field == "expanded_sentence":
                    expand_response = output_data[field]
                    assert isinstance(expand_response, str)
                    return expand_response

                    # print("Converted output to list:")
                    # print(output_data)
                    # print(type(output_data))
            except json.JSONDecodeError as e:
                retry -= 1
                if retry == 0:
                    return {"error": "JSONDecodeError", "message": str(e), "output": output_str}


class RAGASEvalutor:

    def __init__(self,
                 ) -> None:
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini"))
        self.opena_expander = OpenAIExpander()

    def evaluate_answer_accuracy(self, target: str, prediction: str, retry_number=1):
        sample = SingleTurnSample(
            response=prediction,
            reference=target
        )

        score_dict = {"f1": [], "recall": [], "precision": []}
        while retry_number > 0:
            try:
                scores = self.compute_score("f1", sample)

                score_dict["f1"].append(scores["f1"])
                score_dict["recall"].append(scores["recall"])
                score_dict["precision"].append(scores["precision"])
            except Exception as e:
                print(f"Error during evaluation: {e}")
            retry_number -= 1

        for each_metirc, score in score_dict.items():
            if score:
                score_dict[each_metirc] = statistics.median(score)
            else:
                score_dict[each_metirc] = [-1]
        return score_dict

    def evaluate_hallucinations(self, question: str, response: str,  retrieved_contexts: list, retry_number=1):
        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts)

        scorer = Faithfulness()
        scorer.llm = self.evaluator_llm

        score_list = []
        nan_count = 0

        while retry_number > 0:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    score = asyncio.run(scorer.single_turn_ascore(sample))

                    # for warning in w:
                    #     if "No statements were generated" in str(warning.message):
                    #         print(f"Warning caught: {warning.message}")
                    #         expand_response = self.opena_expander.expand_response_to_sentence(
                    #             question, response)
                    #         print(
                    #             f"original response:{response}, expand_response:{expand_response}")
                    #         sample = SingleTurnSample(
                    #             user_input=question,
                    #             response=expand_response,
                    #             retrieved_contexts=retrieved_contexts)
                    #         continue

                    if math.isnan(score):
                        expand_response = self.opena_expander.expand_response_to_sentence(
                            question, response)
                        print(
                            f"original response:{response}, expand_response:{expand_response}")
                        sample = SingleTurnSample(
                            user_input=question,
                            response=expand_response,
                            retrieved_contexts=retrieved_contexts)
                        nan_count += 1
                        if nan_count >= 3:
                            print(
                                f"question: {question}: Encountered 3 consecutive NaN scores, stopping evaluation.")
                            break
                        continue  # 跳过当前循环，继续下一次
                    score_list.append(score)
            except Exception as e:
                print(f"Error during evaluation: {e}")
            retry_number -= 1

        if score_list:
            median_score = statistics.median(score_list)
        else:
            median_score = -1
        return median_score

    def compute_score(self, mode: str, sample: SingleTurnSample):
        scorer = FactualCorrectness(mode=mode,      # 这里修改了 FactualCorrectness的源码，默认只返回一个值，修改后返回f1, recall, precision
                                    atomicity="high", coverage="high")
        scorer.llm = self.evaluator_llm
        score = asyncio.run(scorer.single_turn_ascore(sample))
        return score

    def evaluate_generation_single_question(self, query: str, target: str, prediction: str, retrieval_context: list, eval_metrics=[faithfulness, answer_relevancy, answer_correctness, answer_similarity], retry_number=1):
        '''
            衡量单个问题的生成质量
            指标：
                （1）忠诚度 faithfulness -- 生成的答案是不是能从检索出来的上下文中推出 -> 可用于衡量幻觉性
                （2）答案相关性 answer_relevancy -- 生成的答案是不是能够解决问题
                （3）答案正确性 answer_correctness -- 衡量生成的答案与基本事实相比的准确性（语义相似性和回答的正确性）
                （4）答案相似性 answer_similarity -- 衡量生成的答案与基本事实的语义相似性
                （5）答案表达调整判断   aspect critique -- 有害性、恶意性、连贯性、正确性、简洁性
        '''
        eval_data_sample = {
            'question': [query],
            'answer': [prediction],
            'contexts': [retrieval_context],
            "ground_truth": [target],
        }
        dataset = Dataset.from_dict(eval_data_sample)
        score_list = defaultdict(list)
        nan_count = 0
        max_exception_retries = 1
        while retry_number > 0:
            try:
                scores = evaluate(dataset, metrics=eval_metrics).scores
                for each_metirc in scores:
                    for key, score in each_metirc.items():
                        if math.isnan(score):
                            nan_count += 1
                            if nan_count >= 3:
                                print(
                                    f"question: {question}: Encountered 3 consecutive NaN scores, stopping evaluation.")
                                break
                            continue  # 跳过当前循环，继续下一次
                        score_list[key].append(score)
                retry_number -= 1
            except Exception as e:
                if max_exception_retries > 2:
                    print(f"Error during evaluation: {e}")
                    break
                max_exception_retries += 1
                continue

        # print(score_list)
        for each_metirc, score in score_list.items():
            if score:
                score_list[each_metirc] = statistics.median(score)
            else:
                score_list[each_metirc] = [0]
        # print(score_list)
        if len(score_list) == 1:
            return [float(value) for value in score_list.values()][0]
        elif len(score_list) > 1:
            return score_list
        else:
            return None

    def evaluate_retrieval_single_question(self, query: str, ground_truth: str, actual_output: str, retrieval_context: list):
        '''
            衡量单个问题的生成质量
            指标：
                （1）上下文精度 context_precision -- 与真实答案高度相关的上下文是否出现在rank higher 
                （2）上下文利用率 context_utilization -- 与 context_precision 相同
                （3）上下文召回率 context_recall -- 真实答案中有多少能从检索的上下文中推出来
                （4）上下文相关性 context_relevancy -- 检索出的上下文和问题的相关度，值越大，相关度越高
        '''
        eval_metrics = [context_precision, context_recall]  # context_relevancy
        eval_data_sample = {
            'question': [query],
            'answer': [actual_output],
            'contexts': [retrieval_context],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(eval_data_sample)
        score = evaluate(dataset, metrics=eval_metrics)
        return score


if __name__ == '__main__':

    # data_samples = {
    #     'question': ['What is the capital of France?'],
    #     'answer': ['Paris'],
    #     'contexts' : [['Paris']],
    #     "ground_truth":['Paris'],
    # }
    # dataset = Dataset.from_dict(data_samples)
    # score = evaluate(dataset,metrics=[context_precision, context_utilization])
    # print(score.to_pandas())

    # data_samples = {
    #     'question': ['Where is France and what is it’s capital?'],
    #     'answer': ['Paris'],
    #     'contexts' : [["France, xxxxxMediterranean beaches. ", "xxxx, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower."]],
    #     "ground_truth":['France is in Western Europe and its capital is Paris'],
    # }
    # dataset = Dataset.from_dict(data_samples)
    # score = evaluate(dataset,metrics=[context_recall])

    data_samples2 = {
        'question': ['Where is France and what is it’s capital?'],
        'answer': ['Paris'],
        'contexts': [["France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. ", " The country is also renowned for its wines and sophisticated cuisine.", " Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history."]],
        "ground_truth": ['France is in Western Europe and its capital is Paris'],
    }
    data_samples3 = {
        'question': ["Who were the top two contestants in the America's Got Talent Season 17 finale?"],
        'answer':  ["The top two contestants in the America's Got Talent Season 17 finale were Mayyas and Kristy Sellars. Mayyas won the competition, while Kristy Sellars finished as the runner-up."],
        'contexts': [["Season 17 <-Is finalist in- Kristy sellars -Finished second-> Season 17 of america's got talent", "Season 17 <-Won- Mayyas -Is a finalist-> America's got talent", "Season 17 <-Is winner of- Mayyas -Is a finalist-> America's got talent", "Season 17 <-Has- America's got talent <-Is a finalist- Kristy sellars", "Season 17 <-Has- America's got talent", "Season 17 <-Is- Agt -Is-> America's got talent", "Season 17 <-Is finalist in- Kristy sellars -Became runner-up-> Season 17 of america's got talent", "Season 17 <-Is finalist in- Kristy sellars -Is a finalist-> America's got talent", "Season 17 <-Won- Mayyas -Won-> Season 17 of america's got talent", "Season 17 <-Has- America's got talent <-Is a finalist- Drake milligan", "Season 17 <-Has- America's got talent <-Is a finalist- Avery dixon", "Season 17 <-Has- America's got talent <-Is a finalist- Mayyas", "Season 17 <-Has- America's got talent <-Is a finalist- Celia munoz", "Season 17 <-Has- America's got talent <-Is a finalist- Chapel hart", "Season 17 <-Has- America's got talent <-Is a finalist- Sara james", "Season 17 <-Is winner of- Mayyas -Won-> Season 17 of america's got talent", "Season 17 <-Has- America's got talent <-Is a finalist- Mike e. winfield", "Season 17 <-Has- America's got talent <-Is a finalist- Nicolas ribs", "Season 17 <-Has- America's got talent <-Is a finalist- Metaphysic", "Season 17 <-Has- America's got talent <-Is a finalist- Yu hojin", "Season 17 <-Has- America's got talent <-Is- Agt", "Season 17 <-Has- America's got talent <-Gained recognition- Evancho", "Season 17 <-Has- America's got talent -Was-> Executive producer", "Season 17 <-Has- America's got talent <-Appeared on- Henry winkler", "Season 17 <-Is finalist in- Kristy sellars", "Season 17 -Had-> Two final acts", "Season 17 <-Is winner of- Mayyas -Won-> Arabs got talent", "Season 17 <-Has- America's got talent <-Stands for- Agt", "Season 17 <-Is finalist in- Kristy sellars -Advances to-> Top 5", "Season 17 <-Is- Agt -Stands for-> America's got talent"]],
        "ground_truth": ["Mayyas, Kristy Sellars"],
    }


# M. Young was admitted to Richmond General Hospital on February 28th. She is a female newborn, born at 40 weeks of gestation via normal vaginal delivery, and resides in Richmond. Her birth weight is not mentioned, but she is aged 8 hours after birth. M. Young's parents are healthy, and there is no family history of respiratory or genetic disorders.
    data_samples1 = {
        'question': ["According to the hospitalization records of Richmond General Hospital, summarize the basic information of M. Young."],
        'answer':  ["M. Young is a female newborn, 8 hours after birth, with an unknown ethnicity and marital status, and no occupation. Her address is listed as 90, Maplewood street, Richmond. She was admitted to Richmond General Hospital on February 28th, with her mother serving as the historian for her hospitalization record."],
        'contexts': [[]],
        "ground_truth": ["M. Young is a female newborn, 8 hours after birth, residing at 90, Maplewood street, Richmond. She was admitted to Richmond General Hospital on 28th February, with the record time also on 28th February. The historian is her mother."],
    }
    question = "According to the hospitalization records of Richmond General Hospital, summarize the basic information of M. Young."
    ground_truth2 = "M. Young is a female newborn, 8 hours after birth, residing at 90, Maplewood street, Richmond. She was admitted to Richmond General Hospital on 28th February, with the record time also on 28th February. The historian is her mother."
    actual_output = "M. Young was admitted to Richmond General Hospital on February 28th. She is a female newborn, born at 40 weeks of gestation via normal vaginal delivery, and resides in Richmond. Her birth weight is not mentioned, but she is aged 8 hours after birth. M. Young's parents are healthy, and there is no family history of respiratory or genetic disorders."
    retrieval_contexts2 = []
    ragas_evaluator = RAGASEvalutor()
    score = ragas_evaluator.evaluate_generation_single_question(question,
                                                                ground_truth2,
                                                                actual_output,
                                                                retrieval_contexts2,
                                                                eval_metrics=[answer_correctness])
    print(type(score))

    print(ragas_evaluator.evaluate_answer_accuracy(ground_truth2, actual_output))

    eval_metrics2 = [context_precision, context_recall, context_relevancy]
    eval_metrics = [faithfulness, answer_relevancy,
                    answer_correctness, answer_similarity]
    dataset1 = Dataset.from_dict(data_samples3)
    score1 = evaluate(dataset1, metrics=eval_metrics)
    score2 = evaluate(dataset1, metrics=eval_metrics2)

    # llm = OpenAI("gpt-3.5-turbo")
    # evaluator = CorrectnessEvaluator(llm=llm)
    # result = evaluator.evaluate(
    #     query="Who were the top two contestants in the America's Got Talent Season 17 finale?",
    #     response="The top two contestants in the America's Got Talent Season 17 finale were aaaaa and bbbbbb. aaaaa won the competition, while bbbbbb finished as the runner-up.",
    #     reference=["Season 17 <-Is finalist in- 1"]
    # )
    # # print(score.to_pandas())
    # print(result)
    # print(score1.to_pandas())

    # data_graph_sample_query = "Who were the top two contestants in the America's Got Talent Season 17 finale?"
    # question = "Where is France and what is it’s capital?"
    # ground_truth = ["Mayyas", "Kristy Sellars"]
    # ground_truth2 = ['France is in Western Europe and its capital is Paris']
    # actual_output = ["The top two contestants in the America's Got Talent Season 17 finale were Mayyas and Kristy Sellars. Mayyas won the competition, while Kristy Sellars finished as the runner-up."]
    # retrieval_context = ["Season 17 <-Is finalist in- Kristy sellars -Finished second-> Season 17 of america's got talent", "Season 17 <-Won- Mayyas -Is a finalist-> America's got talent", "Season 17 <-Is winner of- Mayyas -Is a finalist-> America's got talent", "Season 17 <-Has- America's got talent <-Is a finalist- Kristy sellars", "Season 17 <-Has- America's got talent", "Season 17 <-Is- Agt -Is-> America's got talent", "Season 17 <-Is finalist in- Kristy sellars -Became runner-up-> Season 17 of america's got talent", "Season 17 <-Is finalist in- Kristy sellars -Is a finalist-> America's got talent", "Season 17 <-Won- Mayyas -Won-> Season 17 of america's got talent", "Season 17 <-Has- America's got talent <-Is a finalist- Drake milligan", "Season 17 <-Has- America's got talent <-Is a finalist- Avery dixon", "Season 17 <-Has- America's got talent <-Is a finalist- Mayyas", "Season 17 <-Has- America's got talent <-Is a finalist- Celia munoz", "Season 17 <-Has- America's got talent <-Is a finalist- Chapel hart",
    #                      "Season 17 <-Has- America's got talent <-Is a finalist- Sara james", "Season 17 <-Is winner of- Mayyas -Won-> Season 17 of america's got talent", "Season 17 <-Has- America's got talent <-Is a finalist- Mike e. winfield", "Season 17 <-Has- America's got talent <-Is a finalist- Nicolas ribs", "Season 17 <-Has- America's got talent <-Is a finalist- Metaphysic", "Season 17 <-Has- America's got talent <-Is a finalist- Yu hojin", "Season 17 <-Has- America's got talent <-Is- Agt", "Season 17 <-Has- America's got talent <-Gained recognition- Evancho", "Season 17 <-Has- America's got talent -Was-> Executive producer", "Season 17 <-Has- America's got talent <-Appeared on- Henry winkler", "Season 17 <-Is finalist in- Kristy sellars", "Season 17 -Had-> Two final acts", "Season 17 <-Is winner of- Mayyas -Won-> Arabs got talent", "Season 17 <-Has- America's got talent <-Stands for- Agt", "Season 17 <-Is finalist in- Kristy sellars -Advances to-> Top 5", "Season 17 <-Is- Agt -Stands for-> America's got talent"]
    # retrieval_contexts2 = ["season 17"]
    # ragas_evaluator = RAGASEvalutor()
    # ragas_evaluator.evaluate_generation_single_question(question,
    #                                                     ground_truth2,
    #                                                     actual_output,
    #                                                     retrieval_contexts2)
    # ragas_evaluator.evaluate_retrieval_single_question(data_graph_sample_query,
    #                                                    ground_truth,
    #                                                    actual_output,
    #                                                    retrieval_context)

    expander = OpenAIExpander()
    print(expander.expand_response_to_sentence(
        "What is the premiere date of New Amsterdam Season 5?", "September 20, 2022."))


