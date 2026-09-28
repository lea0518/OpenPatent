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

        # 预置为 None：若 requests 本身就抛异常（超时/DNS失败），下面 except 里
        # 引用 response 才不会变成 UnboundLocalError 把真正的网络错误掩盖掉
        response = None
        try:
            response = requests.request(
                "POST",
                "https://api.siliconflow.cn/v1/embeddings",
                headers=headers,
                json={
                    "model": 'BAAI/bge-m3',
                    "input": text,
                    "encoding_format": "float"
                },
                timeout=30,  # 不设超时的话，连接挂住会一直卡死整个建库流程
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
        # 重置为空索引：本方法语义是"重建"而非"追加"。若不清空，同一实例被调用
        # 两次会让向量与 texts 双双翻倍，且 save_index 会把污染后的状态存盘。
        self.index = faiss.IndexFlatL2(1024)
        self.texts = {}
        self.next_index = 0

        embeddings = []
        for text in texts:
            # 跳过空段落：上游某篇 PDF 缺章节时会传进 None，直接送去 embedding
            # 会被 API 以 400 拒绝，导致整批建库失败
            if not text or not str(text).strip():
                logging.warning(f'[{self.db_type}] 跳过一条空文本（上游 PDF 可能缺少该章节）')
                continue
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
            self.texts[self.next_index] = text
            self.next_index += 1

        if not embeddings:
            raise ValueError(f'[{self.db_type}] 没有任何可索引的文本，请检查参考专利 PDF 是否被正确切分')

        print(f"data created: {self.db_type} 共 {len(self.texts)} 条")
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

        抛出:
        FileNotFoundError: 缺少 .texts.json（旧版知识库），检索会取不到原文。
        ValueError: 向量数与文本数不一致，知识库已损坏。
        """
        self.index = faiss.read_index(path)
        # 一并加载文本字典（JSON 的 key 会变成字符串，需转回 int 以匹配 FAISS 返回的整数下标）
        import json
        texts_path = path + ".texts.json"
        # 这里必须硬失败而非 warning：缺 texts 时 FAISS 向量仍在、search() 只是
        # 静默返回空列表，界面照样出稿，RAG 却已空转——这个"不报错的失效"曾污染
        # 阶段0/1/2 的全部实验数据。宁可启动即拒，也不要静默降级。
        if not os.path.exists(texts_path):
            raise FileNotFoundError(
                f"知识库 {path} 缺少文本文件 {texts_path}（旧版知识库只存了向量）。\n"
                f"检索将取不到原文、RAG 会静默失效。请重新点击『处理专利文件』重建知识库。"
            )
        with open(texts_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.texts = {int(k): v for k, v in raw.items()}
        self.next_index = len(self.texts)

        # 一致性校验：两者数量对不上说明库被写坏了（如建库中途失败）
        if self.index.ntotal != len(self.texts):
            raise ValueError(
                f"知识库 {path} 已损坏：向量数 {self.index.ntotal} 与文本数 "
                f"{len(self.texts)} 不一致。请重新点击『处理专利文件』重建知识库。"
            )

    def search(self, query: str, k=2) -> List[str]:
        """
        根据查询文本搜索相关文本。

        参数:
        query (str): 查询文本。
        k (int): 返回的相关文本数量，默认为 2。

        返回:
        List[str]: 相关文本列表。
        """
        print(f"[{self.db_type}] 检索中，query 长度 {len(query)} 字")
        embedding = self._get_embedding(query)
        embedding = embedding.astype('float32')
        D, I = self.index.search(embedding, k) #返回的是 [batchsize,index]形状的数组
        I = np.squeeze(I)
        D = np.squeeze(D)
        try:
            # FAISS 在候选不足 k 个时会用 -1 填充，需剔除
            results = [self.texts[i] for i in np.atleast_1d(I) if i >= 0]
            print(f"[{self.db_type}] 检索命中 {len(results)} 条（下标 {np.atleast_1d(I).tolist()}）")
            return results
        except KeyError as e:
            # 走到这里说明 texts 与索引不匹配，属知识库损坏；load_index 已加校验，
            # 正常流程不该触发。保留兜底但把信息说清，避免又变成静默失效。
            logging.error(f"[{self.db_type}] 检索下标 {e} 在文本字典中不存在，"
                          f"知识库可能已损坏，请重新『处理专利文件』重建。")
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
