# -*- coding: utf-8 -*-
"""
数据清洗模块
============
功能：
1. 文本清洗：去 HTML 标签、压缩空白、过滤特殊字符
2. 数据校验：过滤空标题、正文过短的记录
3. 数据库清理：去重、删除无效记录
4. 数据导出：导出为 CSV（方便用 pandas 分析 / 训练模型）

运行方式：
    python -m crawler.clean            # 清理数据库并打印统计
    python -m crawler.clean --export   # 清理后导出 CSV 到 data/news.csv
"""
import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# 多余空白字符
_WHITESPACE_RE = re.compile(r"[ \t\u3000]+")
# 连续换行
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# HTML 标签（正文提取失败时兜底）
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# 控制字符
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_html(raw):
    """去除 HTML 标签，返回纯文本。"""
    if not raw:
        return ""
    text = _HTML_TAG_RE.sub("", raw)
    return text


def clean_text(text):
    """
    清洗新闻正文/标题：
    - 去 HTML 标签
    - 去控制字符
    - 压缩空白和多余换行
    - 去掉首尾空白
    """
    if not text:
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def is_valid_news(title, content, min_len=config.MIN_CONTENT_LEN):
    """判断一条新闻是否有效：标题非空且正文足够长。"""
    if not title or not content:
        return False
    if len(clean_text(title)) < 4:
        return False
    if len(clean_text(content)) < min_len:
        return False
    return True


def clean_db(db_path=config.DB_PATH):
    """
    对数据库执行清洗：
    1. 删除标题/正文为空的记录
    2. 删除正文过短的记录
    3. 清洗每条记录中的 HTML 标签和空白
    返回 (删除条数, 清洗条数)
    """
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. 删除无效记录
    cur.execute(
        "DELETE FROM news WHERE title IS NULL OR trim(title) = '' "
        "OR content IS NULL OR trim(content) = ''"
    )
    deleted_null = cur.rowcount

    # 2. 读取全部记录，过滤正文过短并清洗
    cur.execute("SELECT id, title, content, source, publish_time FROM news")
    rows = cur.fetchall()

    deleted_short = 0
    updated = 0
    for row in rows:
        news_id, title, content, source, publish_time = row
        clean_title = clean_text(title)
        clean_content = clean_text(content)
        if len(clean_content) < config.MIN_CONTENT_LEN:
            cur.execute("DELETE FROM news WHERE id = ?", (news_id,))
            deleted_short += 1
            continue
        if clean_title != title or clean_content != content:
            cur.execute(
                "UPDATE news SET title = ?, content = ?, source = ?, publish_time = ? WHERE id = ?",
                (clean_title, clean_content, clean_text(source), clean_text(publish_time), news_id),
            )
            updated += 1
    conn.commit()
    conn.close()
    return deleted_null + deleted_short, updated


def export_csv(db_path=config.DB_PATH, csv_path=config.CSV_PATH):
    """把数据库新闻导出为 CSV（列：id,title,content,category,url,source,publish_time）。"""
    import csv

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, content, category, url, source, publish_time FROM news ORDER BY id"
    )
    rows = cur.fetchall()
    conn.close()

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "content", "category", "url", "source", "publish_time"])
        writer.writerows(rows)
    return len(rows)


def show_stats(db_path=config.DB_PATH):
    """打印数据库各分类统计。"""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT category, COUNT(*) FROM news GROUP BY category ORDER BY COUNT(*) DESC")
    rows = cur.fetchall()
    total = sum(r[1] for r in rows)
    conn.close()
    print("数据库分类统计：")
    for cat, cnt in rows:
        print(f"  {cat or '(未分类)':<8} {cnt} 条")
    print(f"  {'合计':<8} {total} 条")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新闻数据清洗")
    parser.add_argument("--export", action="store_true", help="清洗后导出 CSV")
    args = parser.parse_args()

    print("开始清洗数据库...")
    deleted, updated = clean_db()
    print(f"清洗完成：删除无效记录 {deleted} 条，更新清洗 {updated} 条")
    show_stats()

    if args.export:
        n = export_csv()
        print(f"已导出 {n} 条数据到 {config.CSV_PATH}")
