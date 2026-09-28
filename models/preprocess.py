# -*- coding: utf-8 -*-
"""
文本预处理与特征提取
======================
功能：
1. 加载停用词表
2. 中文分词（jieba）
3. 从 SQLite / CSV 加载新闻数据
4. TF-IDF 特征向量化
5. 向量器与标签的保存/加载

运行方式：
    python -m models.preprocess   # 快速自检：打印分词与特征示例
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# jieba 首次运行会加载词典，打印到 stderr 不影响使用
_DEFAULT_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "我", "你", "他", "她", "它", "们",
    "这", "那", "与", "及", "或", "等", "为", "被", "把", "让", "对", "从",
    "向", "到", "于", "中", "上", "下", "而", "但", "就", "都", "也", "还",
    "又", "再", "因为", "所以", "如果", "虽然", "但是", "然而", "于是", "然后",
    "一个", "一种", "一些", "这个", "那个", "这些", "那些", "可以", "可能",
    "能够", "需要", "应该", "没有", "不是", "就是", "还是", "只是", "不要",
    "不会", "不能", "时候", "已经", "正在", "曾经", "目前", "现在", "今天",
    "昨天", "明天", "年", "月", "日", "时", "分", "秒", "记者", "报道", "新闻",
    "表示", "进行", "相关", "其中", "以及", "通过", "根据", "成为", "作为",
    "对于", "关于", "问题", "方面", "情况", "发现", "出现", "认为", "称", "说",
    "据悉", "了解", "网易", "返回", "更多", "http", "https", "www", "com",
    "html", "nbsp", "amp",
}


# ================= 停用词 =================

def load_stopwords(path=config.STOPWORDS_PATH):
    """加载停用词表；文件不存在时使用内置默认停用词。"""
    path = Path(path)
    if path.exists():
        words = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w)
        return words
    return _DEFAULT_STOPWORDS


# ================= 分词 =================

def tokenize(text, stopwords=None):
    """
    对文本分词并过滤停用词。
    返回空格分隔的字符串（供 TfidfVectorizer 使用）。
    """
    if not text:
        return ""
    if stopwords is None:
        stopwords = load_stopwords()
    words = jieba.cut(str(text))
    tokens = [w.strip() for w in words if w.strip() and w not in stopwords and len(w.strip()) > 1]
    return " ".join(tokens)


# ================= 数据加载 =================

def load_news_from_db(db_path=config.DB_PATH):
    """
    从 SQLite 加载新闻数据。
    返回 DataFrame，列：title, content, category。
    """
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT title, content, category FROM news "
        "WHERE category IS NOT NULL AND category != '' AND content IS NOT NULL",
        conn,
    )
    conn.close()
    return df


def load_news_from_csv(csv_path=config.CSV_PATH):
    """
    从 CSV 加载新闻数据（支持 THUCNews 等公开数据集格式）。
    要求包含列：title, content, category（大小写不敏感，label/cat 也可自动识别）。
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 兼容常见列名
    col_map = {"title": "title", "content": "content", "text": "content",
               "label": "category", "cat": "category", "category": "category"}
    df = df.rename(columns=col_map)

    missing = [c for c in ("title", "content", "category") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}，请确保包含 title/content/category（或 label）")
    return df[["title", "content", "category"]]


def load_news():
    """
    自动选择数据源：优先数据库，数据库为空则尝试 CSV。
    """
    df = load_news_from_db()
    if len(df) == 0:
        csv_file = Path(config.CSV_PATH)
        if csv_file.exists():
            print(f"数据库为空，从 CSV 加载: {csv_file}")
            df = load_news_from_csv()
        else:
            print("数据库和 CSV 都没有数据！请先运行: python -m crawler.spider")
    return df


# ================= TF-IDF 特征 =================

def build_tfidf(texts, max_features=config.MAX_FEATURES, min_df=config.MIN_DF):
    """
    训练 TF-IDF 向量器并返回特征矩阵。
    texts: 已分词的文本列表（空格分隔）
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        token_pattern=r"(?u)\b\w+\b",  # 中文按空格分词后的 token
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


# ================= 保存 / 加载 =================

def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # 自检
    print("加载停用词表...")
    sw = load_stopwords()
    print(f"停用词数量: {len(sw)}")

    sample = "网易新闻报道：中国科技公司在人工智能领域取得重大突破。"
    print(f"原文: {sample}")
    print(f"分词: {tokenize(sample, sw)}")

    print("\n尝试加载数据...")
    df = load_news()
    print(f"数据条数: {len(df)}")
    if len(df) > 0:
        print("类别分布:")
        print(df["category"].value_counts())
