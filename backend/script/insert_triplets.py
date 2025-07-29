import argparse

# import json
import json
import multiprocessing
import os
import time
from typing import Dict, List

from database.graph.nebulagraph.nebulagraph import NebulaDB

# /home/lipz/NeutronRAG/NeutronRAG/backend/triplets/rgb_triplets.json

def file_exist(path):
    return os.path.exists(path)
def read_json(file_path: str):
    assert file_exist(file_path), file_path
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_to_json(file_path: str, data, indent=2, info=True):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
    if info:
        print(f"save {len(data)} items to {file_path}")


def read_jsonl(file_path: str) -> List[Dict]:
    assert file_exist(file_path)
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def save_to_jsonl(file_path: str, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            # json.dump(data, file, ensure_ascii=False, indent=2)
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"save {len(data)} items to {file_path}")


# import os
# import sys
# from utils.utils import create_dir
# home_dir = os.path.expanduser('~')

# if log:
#     log_dir = os.path.join(home_dir, 'rag-cache/experiment/exp1-motivation/log')
#     create_dir(log_dir)
#     log_file = os.path.join(log_dir, f'insert-{len(triplets)}-triplets-pid{pid}.log')

# create_dir('./log')
# savedStdout = sys.stdout  #保存标准输出流
# file = open(log_filename, 'w')
# sys.stdout = file  #标准输出重定向至文件

# if args.log_path:
#     file.close()
#     sys.stdout = savedStdout  #标准输出重定向至文件

# from tqdm import tqdm


def insert_triple(pid, triplets, graph_db: NebulaDB, verbose=False, log=False):
    start_time = time.time()
    # for i, triplet in tqdm(enumerate(triplets), f"insert triplets in {db_names}"):
    for i, triplet in enumerate(triplets):
        if i and i % 10000 == 0 and verbose:
            print(
                f"processor {pid} insert {i}/{len(triplets)} triplets, cost {time.time() - start_time : .3f}s."
            )
        graph_db.upsert_triplet(triplet)
    end_time = time.time()
    print(
        f"pid {pid} insert {len(triplets)} triplets cost {end_time - start_time : .3f}s."
    )


def parallel_insert(triplets, db_name, nproc=5, reuse=False):
    start_time = time.time()
    processes = []
    n_triplets = len(triplets)
    if n_triplets < 100:
        nproc = 1
    step = (n_triplets + nproc - 1) // nproc

    print(n_triplets, db_name, nproc, step)
    print(f"\ninsert {n_triplets} triplets in {db_name}, nproc={nproc}")

    nebula_db = NebulaDB(db_name)
    if not reuse:
        nebula_db.clear()

    for i in range(nproc):
        start = i * step
        end = min(start + step, n_triplets)
        print(f"pid {i} take {start}-{end}")
        p = multiprocessing.Process(
            target=insert_triple,
            args=(
                i,
                triplets[start:end],
                nebula_db,
                True,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()

    print(f"insert_triple_parallel cost {end_time - start_time:.3f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", type=str, default=None, help="database name, e.g. test."
    )
    parser.add_argument("--reuse", action="store_true", help="reuse database")
    # parser.add_argument("--data", type=str, required=True, help="dataset name.")
    parser.add_argument(
        "--input", type=str, required=True, help="input triplet json file."
    )
    parser.add_argument(
        "--proc",
        type=int,
        # required=True,
        default=1,
        help="processor numbers, e.g. test.",
    )

    args = parser.parse_args()
    # if not args.db:
    #     args.db = args.data

    print(args)

    # filename = f"../triplet/triplets/{args.data}_triplets.json"
    filename = args.input
    loaded_triplets = read_json(filename)
    loaded_triplets = [(str(x), str(y), str(z)) for x, y, z in loaded_triplets]
    loaded_triplets = list(set(loaded_triplets))

    for triplet in loaded_triplets:
        assert len(triplet) == 3
        for x in triplet:
            assert len(x) > 0, triplet
    print(f"load {len(loaded_triplets)} triplets from {filename}.")

    parallel_insert(loaded_triplets, args.db, args.proc, args.reuse)
