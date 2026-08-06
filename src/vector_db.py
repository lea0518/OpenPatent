import os
import faiss
import requests
from typing import List, Dict
import logging
import numpy as np

class VectorDB:
    """
    向量数据库类，用于创建、保存、加载和查询向量索引。
    """
    def __init__(self, db_type: str):
        """
        初始化向量数据库。

        参数:
        db_type (str): 数据库类型，如 "摘要", "说明书", "权利要求书"。
        """
        self.api_key = os.getenv('SILICONFLOW_API_KEY')
        self.db_type = db_type
        self.index = faiss.IndexFlatL2(1024)  # 假设bge-m3的维度为1024
        self.texts: Dict[int, str] = {}  # Store texts with their corresponding index
        self.next_index = 0
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量。

        参数:
        text (str): 要获取嵌入向量的文本。

        返回:
        np.ndarray: 文本的嵌入向量。
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.request(
                "POST",
                "https://api.siliconflow.cn/v1/embeddings",
                headers=headers,
                json={
                    "model": 'BAAI/bge-m3',
                    "input": text,
                    "encoding_format": "float"
                }
            )
            response.raise_for_status()
            return np.array([response.json()['data'][0]['embedding']]).astype('float32')
        except requests.exceptions.RequestException as e:
            logging.error(f'Embedding生成失败: {str(e)}')
            if response is not None:
                logging.error(f'Response content: {response.text}')
            raise

    def create_index(self, texts: List[str]):
        """
        创建向量索引。

        参数:
        texts (List[str]): 要创建索引的文本列表。
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
            self.texts[self.next_index] = text
            self.next_index += 1
        print(f"data created:{self.texts}")
        embeddings = np.concatenate(embeddings, axis=0).astype('float32')
        self.index.add(embeddings) #必须add numpy array

    def save_index(self, path: str):
        """
        保存向量索引到指定路径。

        参数:
        path (str): 保存索引的路径。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        # 同时保存文本字典（FAISS 只存向量，texts 必须单独存，否则加载后检索取不到原文）
        import json
        with open(path + ".texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)

    def load_index(self, path: str):
        """
        从指定路径加载向量索引。

        参数:
        path (str): 加载索引的路径。
        """
        self.index = faiss.read_index(path)
        # 一并加载文本字典（JSON 的 key 会变成字符串，需转回 int 以匹配 FAISS 返回的整数下标）
        import json
        texts_path = path + ".texts.json"
        if os.path.exists(texts_path):
            with open(texts_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.texts = {int(k): v for k, v in raw.items()}
            self.next_index = len(self.texts)
        else:
            logging.warning(f"未找到文本文件 {texts_path}，检索将取不到原文；请重新『处理专利文件』重建知识库。")

    def search(self, query: str, k=2) -> List[str]:
        """
        根据查询文本搜索相关文本。

        参数:
        query (str): 查询文本。
        k (int): 返回的相关文本数量，默认为 2。

        返回:
        List[str]: 相关文本列表。
        """
        print(f"查询： \n {query}")
        embedding = self._get_embedding(query)
        embedding = embedding.astype('float32')
        D, I = self.index.search(embedding, k) #返回的是 [batchsize,index]形状的数组
        I = np.squeeze(I)
        D = np.squeeze(D)
        print(f"I:{I} type:{type(I)}")
        print(self.texts)
        try:
            results = [self.texts[i] for i in I]
            return results
        except KeyError as e:
            logging.error(f"Index error in search results: {e}")
            return []

    def query(self, query: str, top_k=2) -> List[str]:
        """
        根据查询文本查询相关文本。

        参数:
        query (str): 查询文本。
        top_k (int): 返回的相关文本数量，默认为 2。

        返回:
        List[str]: 相关文本列表。
        """
        return self.search(query, k=top_k)
