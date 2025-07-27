'''
Author: lpz 1565561624@qq.com
Date: 2025-05-07 21:18:48
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-05-07 21:24:52
FilePath: /lipz/NeutronRAG/NeutronRAG/data/multi-hop/ambiguous/multi_entity/merge.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json

first_file = "multihop_1.json"
second_file = "multihop_2.json"

with open("multihop_1.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

# 读取第二个 JSON 文件
with open("multihop_2.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# 合并两个列表
merged_data = data1 + data2

# 写入合并后的数据到新文件
with open("multihop.json", "w", encoding="utf-8") as fout:
    json.dump(merged_data, fout, ensure_ascii=False, indent=2)