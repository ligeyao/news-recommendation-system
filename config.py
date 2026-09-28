# -*- coding: utf-8 -*-
"""
全局配置文件
所有模块统一从这里读取路径和参数，避免散落硬编码。
"""
import os

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.db")                # SQLite 新闻库
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")   # 停用词表
CSV_PATH = os.path.join(DATA_DIR, "news.csv")              # 备用：CSV 数据（可放 THUCNews）

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier.pkl")    # 最优分类模型
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")  # TF-IDF 向量器
LABELS_PATH = os.path.join(MODEL_DIR, "label_names.pkl")        # 类别名称列表
REPORT_PATH = os.path.join(MODEL_DIR, "classification_report.txt")  # 评估报告

# ---------- 新闻分类（与爬虫频道对应）----------
# 键：中文类别名（模型标签）  值：网易新闻频道代码
# 说明：网易“游戏/旅游”频道公开接口已关闭（404），故用数据质量更好的
#       “军事/女性”频道替代；如需调整可自行增删（如把某类注释掉）。
CATEGORIES = {
    "财经": "money",
    "体育": "sports",
    "娱乐": "ent",
    "汽车": "auto",
    "科技": "tech",
    "社会": "shehui",
    "军事": "war",
    "女性": "lady",
}

# ---------- 爬虫参数 ----------
CRAWL_DELAY = 0.5            # 每次请求间隔（秒），礼貌爬取
CRAWL_TIMEOUT = 10           # 单次请求超时（秒）
MAX_NEWS_PER_CATEGORY = 100  # 每个频道最多爬取多少条
MIN_CONTENT_LEN = 20         # 正文少于该长度视为无效新闻

# ---------- 自动抓取（方案C：定时自动抓取 + 手动刷新）----------
AUTO_CRAWL_ENABLED = True    # Web 启动后是否自动抓取
AUTO_CRAWL_INTERVAL = 3600   # 自动抓取间隔（秒），1 小时
AUTO_CRAWL_LIMIT = 30        # 自动抓取时每个频道条数
MANUAL_CRAWL_LIMIT = 50      # 手动点击“立即抓取”时每个频道条数

# ---------- 模型参数 ----------
TEST_SIZE = 0.2              # 测试集比例
RANDOM_STATE = 42            # 随机种子
MAX_FEATURES = 20000         # TF-IDF 最大特征数
MIN_DF = 2                   # 忽略出现次数少于该值的词
MIN_SAMPLES_PER_CLASS = 10   # 类别样本数少于此值则丢弃该类别

# ---------- Web ----------
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
