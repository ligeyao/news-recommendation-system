# -*- coding: utf-8 -*-
"""
基于内容的新闻推荐模块
======================
核心思路：
1. 用 TF-IDF 把每条新闻表示成向量
2. 用户浏览历史 -> 兴趣向量（历史新闻向量的平均）
3. 计算兴趣向量与全库新闻的余弦相似度
4. 返回 Top-N 推荐（排除已读，支持类别多样性）

主要接口：
    recommender = ContentRecommender()
    recommender.recommend_by_history([1, 2, 3], top_n=10)   # 基于浏览历史推荐
    recommender.similar_news(news_id=1, top_n=10)           # 单条新闻的相似新闻
    recommender.recommend_by_text("人工智能 芯片", top_n=10) # 基于文本推荐
    recommender.get_latest_news(10)                          # 冷启动兜底：最新新闻

运行方式：
    python -m recommend.recommender   # 命令行演示（需先有数据）
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from models.preprocess import (  # noqa: E402
    build_tfidf,
    load_pickle,
    load_stopwords,
    tokenize,
)


class ContentRecommender:
    """基于内容的新闻推荐器。"""

    def __init__(self, db_path=config.DB_PATH, vectorizer_path=config.VECTORIZER_PATH):
        self.db_path = str(db_path)
        self.vectorizer_path = str(vectorizer_path)
        self.df = None            # 新闻 DataFrame
        self.vectorizer = None    # TF-IDF 向量器
        self.matrix = None        # 新闻 TF-IDF 矩阵（n_news x n_features，稀疏）
        self.id_to_idx = {}       # 新闻 id -> 矩阵行号
        self.idx_to_id = []       # 矩阵行号 -> 新闻 id
        self._load()

    # ================= 加载与构建 =================

    def _load(self):
        """加载新闻数据、向量器，构建 TF-IDF 矩阵。"""
        self._load_news()
        self._load_vectorizer()
        self._build_matrix()

    def _load_news(self):
        """从 SQLite 加载新闻。"""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"数据库不存在: {self.db_path}\n请先运行: python -m crawler.spider"
            )
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT id, title, content, category, url, publish_time "
            "FROM news WHERE content IS NOT NULL AND content != '' "
            "ORDER BY id",
            conn,
        )
        conn.close()
        if df.empty:
            raise ValueError("数据库中没有新闻数据，请先运行: python -m crawler.spider")
        self.df = df.reset_index(drop=True)
        print(f"[推荐模块] 已加载 {len(self.df)} 条新闻")

    def _load_vectorizer(self):
        """优先加载训练阶段保存的向量器，否则用当前新闻重新训练。"""
        vp = Path(self.vectorizer_path)
        if vp.exists():
            try:
                self.vectorizer = load_pickle(vp)
                print(f"[推荐模块] 使用已保存的向量器: {vp}")
                return
            except Exception as e:
                print(f"[推荐模块] 向量器加载失败({e})，重新训练...")
        self.vectorizer, _ = build_tfidf(self._tokenized_texts())

    def _tokenized_texts(self):
        """构造与训练阶段一致的文本（标题重复两次 + 正文），并分词。"""
        stopwords = load_stopwords()
        texts = (
            self.df["title"].fillna("") + " " +
            self.df["title"].fillna("") + " " +
            self.df["content"].fillna("")
        )
        return texts.map(lambda t: tokenize(t, stopwords))

    def _build_matrix(self):
        """构建新闻 TF-IDF 矩阵和 id 映射。"""
        self.matrix = self.vectorizer.transform(self._tokenized_texts())
        self.idx_to_id = self.df["id"].tolist()
        self.id_to_idx = {int(nid): i for i, nid in enumerate(self.idx_to_id)}

    def refresh(self):
        """数据更新后重新加载。"""
        self._load()

    # ================= 基础查询 =================

    def get_news_by_ids(self, news_ids):
        """按 id 列表返回新闻信息列表（保持传入顺序）。"""
        if not news_ids:
            return []
        df = self.df.set_index("id")
        result = []
        for nid in news_ids:
            if nid in df.index:
                row = df.loc[nid]
                result.append({
                    "id": int(nid),
                    "title": row["title"],
                    "content": row["content"],
                    "category": row["category"],
                    "url": row["url"],
                    "publish_time": row["publish_time"],
                })
        return result

    def get_latest_news(self, top_n=10):
        """冷启动兜底：返回最新入库的新闻（按 id 倒序）。"""
        df = self.df.sort_values("id", ascending=False).head(top_n)
        return df.to_dict("records")

    # ================= 推荐核心 =================

    def _rank_by_similarity(self, query_vector, top_n=10, exclude_idxs=None, diversify=True):
        """
        计算 query_vector 与全库新闻的余弦相似度，返回 Top-N 的 (行号, 相似度)。
        diversify=True 时做类别多样性：取 2*top_n 候选后按类别轮询挑选。
        """
        sims = cosine_similarity(query_vector, self.matrix).flatten()
        if exclude_idxs:
            sims[list(exclude_idxs)] = -1.0

        n_candidates = min(len(sims), top_n * 2 if diversify else top_n)
        if n_candidates == 0:
            return []

        if not diversify:
            order = np.argsort(-sims)[:top_n]
            return [(int(i), float(sims[i])) for i in order]

        # 多样性挑选：候选池按相似度排序，轮询各分类取前 top_n
        order = np.argsort(-sims)[:n_candidates]
        by_category = {}
        for i in order:
            cat = self.df.loc[i, "category"]
            by_category.setdefault(cat, []).append(i)

        picked = []
        while len(picked) < top_n and by_category:
            for cat in list(by_category.keys()):
                if len(picked) >= top_n:
                    break
                picked.append(int(by_category[cat].pop(0)))
                if not by_category[cat]:
                    del by_category[cat]
        # 集合通过轮询保证多样性，展示顺序仍按相似度降序
        picked.sort(key=lambda i: sims[i], reverse=True)
        return [(i, float(sims[i])) for i in picked]

    @staticmethod
    def _pack(rows, scores):
        """把 DataFrame 行和相似度打包成统一的推荐结果格式。"""
        items = []
        for row, score in zip(rows, scores):
            d = row.to_dict()
            d["id"] = int(d["id"])
            d["score"] = round(float(score), 4)
            items.append(d)
        return items

    def recommend_by_history(self, history_ids, top_n=10, exclude_history=True, diversify=True):
        """
        根据用户浏览历史（新闻 id 列表）推荐。
        - exclude_history: 是否排除已浏览的新闻
        - diversify: 是否做类别多样性
        """
        valid_ids = [int(i) for i in history_ids if int(i) in self.id_to_idx]
        if not valid_ids:
            print("[推荐模块] 无有效浏览历史，返回最新新闻兜底")
            return self.get_latest_news(top_n)

        idxs = [self.id_to_idx[i] for i in valid_ids]
        # 兴趣向量 = 历史新闻向量的平均（转为 array，新版 sklearn 不支持 np.matrix）
        interest_vec = np.asarray(self.matrix[idxs].mean(axis=0))

        exclude = set(idxs) if exclude_history else set()
        picked = self._rank_by_similarity(
            interest_vec, top_n=top_n, exclude_idxs=exclude, diversify=diversify
        )
        rows = [self.df.loc[i] for i, _ in picked]
        scores = [s for _, s in picked]
        return self._pack(rows, scores)

    def similar_news(self, news_id, top_n=10):
        """单条新闻的相似新闻（用于详情页“相关推荐”），排除自身。"""
        news_id = int(news_id)
        if news_id not in self.id_to_idx:
            return []
        idx = self.id_to_idx[news_id]
        query_vec = self.matrix[idx]
        picked = self._rank_by_similarity(
            query_vec, top_n=top_n, exclude_idxs={idx}, diversify=True
        )
        rows = [self.df.loc[i] for i, _ in picked]
        scores = [s for _, s in picked]
        return self._pack(rows, scores)

    def recommend_by_text(self, text, top_n=10, diversify=True):
        """根据一段文本（如搜索词）推荐最相关的新闻。"""
        if not text or not text.strip():
            return self.get_latest_news(top_n)
        stopwords = load_stopwords()
        tokens = tokenize(text, stopwords)
        if not tokens:
            return self.get_latest_news(top_n)
        query_vec = self.vectorizer.transform([tokens])
        picked = self._rank_by_similarity(query_vec, top_n=top_n, diversify=diversify)
        rows = [self.df.loc[i] for i, _ in picked]
        scores = [s for _, s in picked]
        return self._pack(rows, scores)


# ================= 命令行演示 =================

def _demo():
    """命令行演示推荐效果（需先爬取数据）。"""
    rec = ContentRecommender()

    print("\n===== 最新新闻（冷启动兜底）=====")
    for item in rec.get_latest_news(5):
        print(f"  [{item['id']}] ({item['category']}) {item['title'][:30]}")

    # 取一条科技新闻作为浏览历史，测试推荐
    tech = rec.df[rec.df["category"] == "科技"].head(3)
    if len(tech) == 0:
        tech = rec.df.head(3)
    history = tech["id"].tolist()
    print(f"\n===== 模拟用户浏览历史: {history} =====")
    for item in rec.get_news_by_ids(history):
        print(f"  看过: ({item['category']}) {item['title'][:30]}")

    print("\n===== 基于历史的推荐 Top 10 =====")
    for item in rec.recommend_by_history(history, top_n=10):
        print(f"  [{item['id']}] ({item['category']}) 相似度{item['score']:.3f} {item['title'][:30]}")

    print("\n===== 单条新闻的相似新闻 =====")
    for item in rec.similar_news(history[0], top_n=5):
        print(f"  [{item['id']}] ({item['category']}) 相似度{item['score']:.3f} {item['title'][:30]}")

    print("\n===== 文本推荐：'人工智能 芯片' =====")
    for item in rec.recommend_by_text("人工智能 芯片", top_n=5):
        print(f"  [{item['id']}] ({item['category']}) 相似度{item['score']:.3f} {item['title'][:30]}")


if __name__ == "__main__":
    _demo()
