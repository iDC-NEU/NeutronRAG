<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="./figure/log.png" width="40" height="40" alt="log" style="vertical-align: middle; margin-right: 10px;"/>
  <h2 style="display: inline; vertical-align: middle;">NeutronRAG: Towards Understanding Vector-based RAG and Graph-based RAG</h2>
</div>

<p align="center">
  <img src="./figure/architecture.png" width="300" alt="architecture"/>
</p>
<p align="center">Overall architecture of NeutronRAG</p>




# 🔨Setup

```bash


# Create conda environment: python >= 3.10
conda create --name llmrag python=3.10.14 -y

conda activate llmrag

# Install required Python packages:
pip install -r requirements.txt

```

# Test if OLlama is available:
```bash
ollama run llama2:7b
```


📦 Deploy Graph Database
1. NebulaGraph Installation Guide
Step 1: Install docker-compose
Ensure that you have docker-compose installed. If not, you can install it with the following command:

```bash
sudo apt install docker-compose
```
Step 2: Clone NebulaGraph Docker Compose Repository
In a directory of your choice, clone the NebulaGraph Docker Compose files:

```bash
git clone https://github.com/vesoft-inc/nebula-docker-compose.git
cd nebula-docker-compose
```
Step 3: Start NebulaGraph
In the nebula-docker-compose directory, run the following command to start NebulaGraph:

```bash
docker-compose up -d
```
Step 4: Check NebulaGraph Container Status
After starting, you can verify that the NebulaGraph container is running by using:

```bash
docker ps
```
Step 5: Connect to NebulaGraph
To connect to NebulaGraph inside the container, use the following command:

```bash
nebula-console -u <user> -p <password> --address=graphd --port=9669
#Replace <user> and <password> with the actual username and password. Ensure that port 9669 is used for the default configuration.
```
Step 6: Enable Data Persistence
To ensure that data persists even after the container is restarted, you can mount persistent volumes. Either modify the volumes section in the docker-compose.yaml file, or manually run the following command with specified persistence paths:

```bash
docker run -d --name nebula-graph \
    -v /yourpath/nebula/data:/data \
    -v /yourpath/nebula/logs:/logs \
    -p 9669:9669 \
    vesoft/nebula-graphd:v2.5.0
#Replace /yourpath/nebula with your actual data persistence path.
```




2. Neo4j (Installation optional for now)






# 💄Run 
```
# Configure temporary environment variables
Example export PYTHONPATH=$PYTHONPATH:/home/lipz/RAGWebUi/RAGWebUi_demo/backend
export PYTHONPATH=$PYTHONPATH:/your/path/backend

```

```
# Run a WebUI to display the frontend interface: 
python webui_chat.py

# Use another terminal to run a graph-based UI to display topology in the frontend:
python graph.py
```

```
# Running the backend mainly for research purposes:
python backend_chat.py --dataset_name "rgb" --llm "llama2:7b" --func "Graph RAG" --graphdb "nebulagraph" --vectordb "MilvusDB"
```

# Notion

1. .env file loading is deprecated. Now uses client input, including LLM name
2. The method low_chat() in ./llmragenv/llmrag_env.py is a simplified input version where the LLM name, database usage, etc., are hardcoded. The web_chat method is the full version.
3. LLM support: The llm_provider dictionary in llm_factory lists all currently supported local models. (Commercial model API keys are not enabled here due to cost, but users can purchase them separately and configure in ./config/config-local.yaml.)
4. Frontend ports and database configurations can be modified in ./config/config-local.yaml (vector DB and NebulaGraph are hardcoded in the code, and need refactoring)
5. Code structure:
![avatar](./resource/codestruc/codestruc.bmp)


# Issue:
Although web_chat() allows selecting a different LLM for each chat session, in practice, only the first selection takes effect—subsequent interactions always use the initially chosen model.
# Code Structure:
Includes graphrag, vectorrag, hybridrag




# Reference
[Meet-libai from BinNong](https://github.com/BinNong/meet-libai)
