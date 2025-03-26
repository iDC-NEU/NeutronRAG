'''
Author: lpz 1565561624@qq.com
Date: 2025-03-26 15:33:58
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-26 15:34:06
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/evaluator/dbname_map.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
available_graphdbname = {
    "rgb": "rgb",
    "multihop": "multihop_ccy",
    "arxiv": "arxiv",
    "govreport": "govreport",
    "dragonball": "dragonball",
    "integrationrgb": "integrationrgb",
    "crudrag": "crudrag",
    "hotpotqa": "hotpotqa"
}

available_vectordbname = {
    "rgb": "rgb",
    "multihop": "multihop",
    "arxiv": "arxiv",
    "govreport": "usreport",
    "dragonball": "dragonball",
    "integrationrgb": "integrationrgb",
    "crudrag": "crudrag",
    "hotpotqa": "newhotpotqa"
    # "hotpotqa": "hotpotqa",
}


datatype_map = {
    "rgb": "qa",
    "multihop": "qa",
    "integrationrgb": "qa",
    "hotpotqa": "qa",
    "arxiv": "summary",
    "govreport": "summary",
    "dragonball": "summary",
    "crudrag": "summary",
}


def get_database_name(data_name: str, mode: str):
    if data_name not in available_graphdbname:
        raise ValueError(
            "Error: data  is not valid. It must be one of the following: rgb, multihop, arxiv, govreport.")
    if mode == "vector":
        return available_vectordbname[data_name]
    elif mode == "graph":
        return available_graphdbname[data_name]
    else:
        raise ValueError(
            "Error: mode is not valid. It must be one of the following: graph, vector.")


def get_data_type(data_name: str):
    if data_name in datatype_map:
        return datatype_map[data_name]
    else:
        raise ValueError(
            "Error: data name is not valid. It must be one of the following: rgb, multihop, arxiv, govreport.")


if __name__ == "__main__":
    data_name = "rgb"
    print(get_database_name(data_name, 'graph'))  # Output: rgb
