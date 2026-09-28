# -*- coding: utf-8 -*-
"""
网易新闻爬虫
============
功能：
1. 从网易新闻各频道抓取新闻列表（标题、URL、时间）
2. 进入详情页提取正文
3. 清洗后写入 SQLite（data/news.db）

数据源说明：
- 首选：网易新闻公开 JSON 接口 https://temp.163.com/special/00804KVA/cm_<频道代码>.js
- 备选：频道 HTML 页面 https://news.163.com/<频道代码>/
- 若接口全部失效（网易时常改版），课题允许改用公开数据集（如 THUCNews），
  只需将数据整理为 CSV（列：title, content, category）放到 data/news.csv 即可训练模型。

运行方式：
    python -m crawler.spider            # 爬取所有配置频道
    python -m crawler.spider --cat 财经  # 只爬某个频道
    python -m crawler.spider --limit 50  # 每个频道最多 50 条
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# 允许直接运行和作为模块运行两种方式
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------- 请求头 ----------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "close",
}

API_TEMPLATE = "https://temp.163.com/special/00804KVA/cm_{code}.js"
HTML_TEMPLATE = "https://www.163.com/{code}/"


# ================= 工具函数 =================

def fetch(url, timeout=config.CRAWL_TIMEOUT, retry=2):
    """发送 GET 请求，带重试机制。失败返回 None。"""
    for i in range(retry + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding or "utf-8"
                return resp
            print(f"  [!] HTTP {resp.status_code} 请求失败: {url}")
        except requests.RequestException as e:
            print(f"  [!] 请求异常(第{i + 1}次): {e}")
        time.sleep(config.CRAWL_DELAY * (i + 1))
    return None


def parse_jsonp(text):
    """
    把 JSONP 文本解析成 Python 对象，兼容两种常见格式：
      1. 函数调用式：data_callback([...])
      2. 变量赋值式：var data_callback = [...] 或 var xxx = {...}
    """
    if not text:
        return None
    # 函数调用式：xxx([...]) / xxx({...})
    match = re.search(r"[\w$.]+\s*\(\s*(\[.*\]|\{.*\})\s*\)\s*;?\s*$", text, re.S)
    if not match:
        # 变量赋值式：var xxx = [...] / xxx = {...}
        match = re.search(r"=\s*(\[.*\]|\{.*\})", text, re.S)
    json_text = match.group(1) if match else text
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


# ================= 新闻列表获取 =================

def get_channel_news_via_api(code, limit):
    """
    方式一：网易公开 JSON 接口获取频道新闻列表。
    返回 [{title, url, time}, ...]
    """
    url = API_TEMPLATE.format(code=code)
    print(f"  [*] 尝试接口: {url}")
    resp = fetch(url)
    if resp is None:
        return []

    data = parse_jsonp(resp.text)
    if not isinstance(data, list):
        print("  [!] 接口返回格式异常")
        return []

    items = []
    for row in data:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or row.get("标题")
        docurl = row.get("docurl") or row.get("url") or row.get("link")
        if not title or not docurl:
            continue
        pub_time = row.get("time") or row.get("ptime") or ""
        items.append({"title": title, "url": docurl, "time": pub_time})
        if len(items) >= limit:
            break
    return items


def get_channel_news_via_html(code, limit):
    """
    方式二：解析频道 HTML 页面获取新闻链接（接口失效时兜底）。
    优先取本频道文章链接（/game/article/...），其次取网易号 dy 链接。
    """
    url = HTML_TEMPLATE.format(code=code)
    print(f"  [*] 尝试页面: {url}")
    resp = fetch(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    # 链接匹配优先级：本频道 article > dy/news article > 老式 news 数字链接
    patterns = [
        re.compile(rf"163\.com/{re.escape(code)}/article/[\w]+\.html"),
        re.compile(r"163\.com/(?:dy|news)/article/[\w]+\.html"),
        re.compile(r"news\.163\.com/\d{2}/\d{4}/\d{2}/[\w-]+\.html"),
    ]

    def clean_url(href):
        """去掉 ?clickfrom=... 等 query 参数。"""
        return href.split("?")[0]

    picked = {0: [], 1: [], 2: []}  # 按优先级分组
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        for idx, pat in enumerate(patterns):
            if pat.search(href):
                href = clean_url(href)
                if href in seen:
                    break
                title = a.get_text(strip=True)
                if len(title) < 8:  # 过滤无效短标题
                    break
                seen.add(href)
                picked[idx].append({"title": title, "url": href, "time": ""})
                break

    items = []
    for idx in range(3):  # 高优先级先取
        items.extend(picked[idx])
        if len(items) >= limit:
            break
    return items[:limit]


def get_channel_news(code, limit):
    """获取某个频道的新闻列表，自动选择可用数据源。"""
    items = get_channel_news_via_api(code, limit)
    if not items:
        print("  [*] 接口无数据，切换到 HTML 页面解析...")
        items = get_channel_news_via_html(code, limit)
    return items


# ================= 正文提取 =================

def extract_article(url):
    """
    提取新闻详情：标题、正文、发布时间、来源。
    返回 dict 或 None。
    """
    resp = fetch(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # 标题
    title = ""
    for sel in ["h1", "div.post_content_main h1", "div#h1title", "title"]:
        tag = soup.select_one(sel)
        if tag:
            title = tag.get_text(strip=True)
            if title:
                break
    # title 标签里通常带“_网易新闻”后缀
    if title.endswith("_网易新闻"):
        title = title[:-5]

    # 正文（网易页面结构多样，逐个尝试）
    content = ""
    selectors = [
        "div.post_body",       # 新版网易号/新闻
        "div#content",         # 老版新闻正文
        "div#endText",         # 老版
        "div.article-content",
        "article",
    ]
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            # 去掉脚本、样式、广告
            for junk in tag(["script", "style", "ins"]):
                junk.decompose()
            content = tag.get_text(separator="\n", strip=True)
            if len(content) >= config.MIN_CONTENT_LEN:
                break
            content = ""

    # 发布时间
    pub_time = ""
    meta = soup.find("meta", attrs={"name": re.compile("pubdate|publishdate|date", re.I)})
    if meta and meta.get("content"):
        pub_time = meta["content"].strip()
    else:
        t = soup.select_one("div.post_info, span.post_info, div.article_info, span.time")
        if t:
            pub_time = t.get_text(strip=True)

    # 来源
    source = ""
    src_tag = soup.select_one("a.source, span.source, div.post_info a, #ne_article_source")
    if src_tag:
        source = src_tag.get_text(strip=True)

    if not title or len(content) < config.MIN_CONTENT_LEN:
        return None
    return {"title": title, "content": content, "time": pub_time, "source": source}


# ================= 数据库 =================

class NewsDB:
    """SQLite 新闻数据库操作封装。"""

    def __init__(self, db_path=config.DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    title        TEXT NOT NULL,
                    content      TEXT,
                    category     TEXT,          -- 分类标签（频道名）
                    url          TEXT UNIQUE,   -- 去重依据
                    source       TEXT,
                    publish_time TEXT,
                    crawl_time   TEXT
                )
                """
            )

    def exists(self, url):
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM news WHERE url = ?", (url,))
            return cur.fetchone() is not None

    def insert(self, item):
        """插入一条新闻，url 重复则忽略。返回是否插入成功。"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO news
                    (title, content, category, url, source, publish_time, crawl_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["title"],
                    item["content"],
                    item.get("category", ""),
                    item.get("url", ""),
                    item.get("source", ""),
                    item.get("time", ""),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            return cur.rowcount > 0

    def count(self, category=None):
        with self._connect() as conn:
            if category:
                cur = conn.execute("SELECT COUNT(*) FROM news WHERE category = ?", (category,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM news")
            return cur.fetchone()[0]


# ================= 主流程 =================

def crawl_category(category, code, limit, db):
    """爬取单个频道的新闻。"""
    print(f"\n===== 爬取频道: {category} ({code}) =====")
    news_list = get_channel_news(code, limit)
    print(f"  获取到 {len(news_list)} 条列表新闻")
    if not news_list:
        print("  [!] 该频道未获取到任何新闻，跳过")
        return 0

    ok = 0
    for i, item in enumerate(news_list, 1):
        url = item["url"]
        if db.exists(url):
            print(f"  [{i}/{len(news_list)}] 已存在，跳过: {item['title'][:20]}")
            continue

        print(f"  [{i}/{len(news_list)}] 抓取正文: {item['title'][:20]}...")
        article = extract_article(url)
        time.sleep(config.CRAWL_DELAY)

        if article is None:
            print("    [!] 正文提取失败，跳过")
            continue

        article.update({"url": url, "category": category})
        if db.insert(article):
            ok += 1
            print(f"    [OK] 入库成功 ({article['title'][:20]}...)")
        if ok >= limit:
            break
    print(f"  频道 {category} 完成，新增 {ok} 条")
    return ok


def crawl_all(limit=None):
    """按 config.CATEGORIES 爬取全部频道。"""
    limit = limit or config.MAX_NEWS_PER_CATEGORY
    db = NewsDB()
    total = 0
    for category, code in config.CATEGORIES.items():
        try:
            total += crawl_category(category, code, limit, db)
        except Exception as e:  # 单频道失败不影响整体
            print(f"  [!] 频道 {category} 爬取出错: {e}")
    print(f"\n全部完成，本次共新增 {total} 条新闻，数据库总计 {db.count()} 条")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网易新闻爬虫")
    parser.add_argument("--cat", type=str, default="", help="只爬某个类别，如：财经")
    parser.add_argument("--limit", type=int, default=None, help="每个频道最大条数")
    args = parser.parse_args()

    if args.cat:
        if args.cat not in config.CATEGORIES:
            print(f"未知类别，可选：{list(config.CATEGORIES.keys())}")
            sys.exit(1)
        db = NewsDB()
        crawl_category(args.cat, config.CATEGORIES[args.cat], args.limit or config.MAX_NEWS_PER_CATEGORY, db)
        print(f"\n数据库当前总计: {db.count()} 条")
    else:
        crawl_all(args.limit)
