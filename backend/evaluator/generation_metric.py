from rouge_score import rouge_scorer
from .pruning import *


class QAGenerationEvalutor:

    def __init__(self,
                 ) -> None:
        pass

    def checkanswer(self, prediction, ground_truth, verbose=False):
        ''''
            只要预测字符串包含 ground_truth 列表中的任何一个元素，flag 就会被设置为 True。  
            如果 ground_truth 只有一个元素，则只需检查预测字符串是否包含这个单一元素
        '''
        prediction = prediction.lower()
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        labels = []
        for instance in ground_truth:
            flag = True
            if isinstance(instance, list):
                flag = False
                instance = [i.lower() for i in instance]
                for i in instance:
                    if i in prediction:
                        flag = True
                        break
            else:
                instance = instance.lower()
                if instance not in prediction:
                    flag = False
            labels.append(int(flag))
        return 0 not in labels
        # return labels

    def exact_match(self, prediction, ground_truth):
        """
            Calculate the exact match between the prediction and the ground truth.

            :param prediction: Model's prediction, can be a string or a list of strings
            :param ground_truth: Ground truth, can be a string or a list of strings
            :return: Boolean (if completely matched) or match ratio (float)
        """

        # If not a list, convert to list
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]

        # Convert all strings in the lists to lowercase
        prediction = [str(p).lower() for p in prediction]
        ground_truth = [str(g).lower() for g in ground_truth]

        # Calculate the number of matching elements
        match_count = sum(1 for p in prediction if p in ground_truth)

        # If completely matched, return True; otherwise, return the match ratio
        if match_count == len(ground_truth) and len(prediction) == len(ground_truth):
            return 1
        else:
            return match_count / len(ground_truth)


class SummaryGenerationEvalutor:

    def __init__(self,
                 ) -> None:
        pass

    def get_rougeL_score(self, target, prediction):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(target, prediction)
        return scores['rougeL'].fmeasure

    def embedding_comparision(self, prediction, ground_truth):
        prediction_embedding = np.array(
            get_text_embedding(prediction)).reshape(1, -1)
        ground_truth_embedding = np.array(
            get_text_embedding(ground_truth)).reshape(1, -1)
        similarity = cosine_similarity_np(
            prediction_embedding, ground_truth_embedding)[0]
        print(similarity)
        return similarity


if __name__ == "__main__":
    prediction = ["我是谁", "谁是我"]
    ground_truth = ["我是谁", "我是谁"]

    predictions1 = "The premiere of 'Carole King & James Taylor: Just Call Out My Name' is on January 2, 2022."
    ground_truths1 = [["January 2 2022", "Jan 2, 2022", "Jan. 2, 2022", "January 2, 2022",
                       "2 January 2022", "2 Jan, 2022", "2 Jan., 2022", "2 January, 2022"]]

    print(QAGenerationEvalutor().checkanswer(predictions1, ground_truths1))

    output_reference = "the cat is on the mat"
    target_reference = "the cat sat on the mat"
    print(SummaryGenerationEvalutor().get_rougeL_score(
        output_reference, target_reference))
