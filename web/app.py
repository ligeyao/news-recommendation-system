# -*- coding: utf-8 -*-
"""
新闻推荐系统 Web 服务
====================
页面：
1. /                  最新新闻列表（分页 + 分类统计）
2. /category/<name>   分类新闻列表（分页）
3. /news/<id>         新闻详情（自动记录浏览历史，展示相似推荐）
4. /recommend         个性化推荐（基于浏览历史，可清空历史）
5. /evaluate          模型评估报告可视化
6. /search?q=         文本搜索推荐

运行方式：
    python -m web.app
    浏览器访问 http://127.0.0.1:5000
"""
import math
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from crawler.scheduler import crawl_manager  # noqa: E402
from flask import (  # noqa: E402
    Flask, jsonify, redirect, render_template, request, session, url_for,
)

app = Flask(__name__)
app.secret_key = "news-recommender-course-design"  # 课程设计用途，生产环境请替换

_recommender = None  # 推荐器单例（懒加载）
PAGE_SIZE = 20       # 列表每页条数


# ================= 基础设施 =================

def get_recommender():
    """懒加载推荐器（首次构建 TF-IDF 矩阵需要几秒）。"""
    global _recommender
    if _recommender is None:
        from recommend.recommender import ContentRecommender
        _recommender = ContentRecommender()
    return _recommender


def refresh_recommender():
    """抓取完成后刷新推荐器（若已加载过）。"""
    global _recommender
    if _recommender is not None:
        try:
            _recommender.refresh()
            print("[Web] 推荐器已刷新")
        except Exception as e:
            print(f"[Web] 推荐器刷新失败: {e}")


def get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query_news_page(where="", params=(), page=1, page_size=PAGE_SIZE):
    """
    分页查询新闻列表（含摘要）。
    返回 (rows, total, page, pages)。
    """
    conn = get_db()
    total = conn.execute(
        f"SELECT COUNT(*) AS cnt FROM news {where}", params
    ).fetchone()["cnt"]
    pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, pages))
    offset = (page - 1) * page_size
    cur = conn.execute(
        f"SELECT id, title, category, url, publish_time, "
        f"substr(content, 1, 120) AS summary "
        f"FROM news {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, page_size, offset),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows, total, page, pages


def get_category_stats():
    """获取各分类新闻数量统计（按数量降序）。"""
    conn = get_db()
    cur = conn.execute(
        "SELECT category, COUNT(*) AS cnt FROM news "
        "WHERE category IS NOT NULL AND category != '' "
        "GROUP BY category ORDER BY cnt DESC"
    )
    stats = [{"category": r["category"], "count": r["cnt"]} for r in cur.fetchall()]
    conn.close()
    return stats


def get_total_count():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM news").fetchone()["cnt"]
    conn.close()
    return total


def get_categories():
    """获取配置中的分类名（Web 展示用）。"""
    return list(config.CATEGORIES.keys())


def record_history(news_id):
    """把浏览的新闻 id 记入 session，最新在前，最多保留 50 条。"""
    history = session.get("history", [])
    history = [h for h in history if h != news_id]
    history.insert(0, news_id)
    session["history"] = history[:50]


# ================= 页面路由 =================

@app.route("/")
def index():
    page = request.args.get("page", 1, type=int)
    news, total, page, pages = query_news_page(page=page)
    return render_template(
        "index.html", news=news, categories=get_categories(),
        stats=get_category_stats(), total=total,
        active=None, page_title="最新新闻",
        page=page, pages=pages, base_url="/",
    )


@app.route("/category/<name>")
def category(name):
    page = request.args.get("page", 1, type=int)
    news, total, page, pages = query_news_page(
        "WHERE category = ?", (name,), page=page
    )
    return render_template(
        "index.html", news=news, categories=get_categories(),
        stats=get_category_stats(), total=total,
        active=name, page_title=f"{name}新闻",
        page=page, pages=pages, base_url=f"/category/{name}",
    )


@app.route("/news/<int:news_id>")
def news_detail(news_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM news WHERE id = ?", (news_id,)).fetchone()
    conn.close()
    if row is None:
        return render_template(
            "404.html", categories=get_categories(), news_id=news_id,
        ), 404
    news = dict(row)
    record_history(news_id)

    similar = []
    try:
        similar = get_recommender().similar_news(news_id, top_n=5)
    except Exception as e:
        print(f"[Web] 相似推荐失败: {e}")

    return render_template(
        "detail.html", news=news, similar=similar,
        categories=get_categories(),
    )


@app.route("/recommend")
def recommend_page():
    history = session.get("history", [])
    rec = get_recommender()
    items = rec.recommend_by_history(history, top_n=20)
    history_news = rec.get_news_by_ids(history[:10])
    return render_template(
        "recommend.html", items=items, history_news=history_news,
        categories=get_categories(), history_count=len(history),
    )


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """清空浏览历史，重新开始推荐。"""
    session.pop("history", None)
    return redirect(url_for("recommend_page"))


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    items = []
    if q:
        try:
            items = get_recommender().recommend_by_text(q, top_n=20)
        except Exception as e:
            print(f"[Web] 搜索推荐失败: {e}")
    return render_template(
        "recommend.html", items=items, history_news=[],
        categories=get_categories(), page_title=f"搜索：{q}",
        search_q=q,
    )


@app.route("/evaluate")
def evaluate():
    report_text = ""
    report_path = Path(config.REPORT_PATH)
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
    has_img = (Path(config.MODEL_DIR) / "confusion_matrix.png").exists()
    return render_template(
        "evaluate.html", report=report_text, has_img=has_img,
        categories=get_categories(),
    )


@app.route("/static/confusion_matrix.png")
def confusion_matrix_img():
    """混淆矩阵图片保存在 models/saved/ 下，通过该路由提供访问。"""
    from flask import send_from_directory
    return send_from_directory(config.MODEL_DIR, "confusion_matrix.png")


# ================= 实时抓取接口 =================

@app.route("/crawl", methods=["POST"])
def crawl_now():
    """手动触发一次抓取（每频道 config.MANUAL_CRAWL_LIMIT 条），完成后刷新推荐器。"""
    crawl_manager.start(
        limit=config.MANUAL_CRAWL_LIMIT, callback=refresh_recommender,
    )
    return redirect(request.referrer or url_for("index"))


@app.route("/crawl_status")
def crawl_status():
    """前端轮询抓取状态。"""
    return jsonify(crawl_manager.get_status())


if __name__ == "__main__":
    # 启动后台定时自动抓取（daemon 线程，随 Web 退出而结束）
    if config.AUTO_CRAWL_ENABLED:
        crawl_manager.start_auto(callback=refresh_recommender)
    # use_reloader=False：避免 debug 热重载导致后台线程重复启动
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=True, use_reloader=False)
