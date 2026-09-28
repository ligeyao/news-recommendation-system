# -*- coding: utf-8 -*-
"""
爬虫调度器
==========
方案C：定时自动抓取 + 手动触发抓取
- CrawlManager.start():      手动/定时触发一次抓取（后台线程执行，不阻塞 Web）
- CrawlManager.start_auto(): 启动后台定时循环：Web 启动 8 秒后先爬一轮，之后每隔
                             config.AUTO_CRAWL_INTERVAL 秒再爬一轮
- CrawlManager.get_status(): 返回抓取状态（是否运行中、上次抓取结果）

线程安全：同一时间只允许一个抓取任务在跑，重复触发会被忽略。
"""
import threading
import time
from datetime import datetime

import config
from crawler.spider import crawl_all


class CrawlManager:
    """后台抓取任务管理器（单例使用）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._last_result = None  # {time, added, limit, error}

    # ---------- 状态 ----------

    @property
    def is_running(self):
        with self._lock:
            return self._running

    def get_status(self):
        """返回给前端轮询的状态字典。"""
        with self._lock:
            return {
                "running": self._running,
                "last_result": self._last_result,
            }

    # ---------- 触发抓取 ----------

    def start(self, limit=None, callback=None):
        """
        触发一次抓取（非阻塞）。
        - limit: 每个频道抓取条数，默认取 config.MANUAL_CRAWL_LIMIT
        - callback: 抓取完成后回调（用于刷新推荐器）
        返回 True 表示已启动，False 表示已有任务在运行。
        """
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._last_result = self._last_result  # 保留上次结果

        limit = limit or config.MANUAL_CRAWL_LIMIT

        def _work():
            added, error = 0, None
            try:
                added = crawl_all(limit)
            except Exception as e:
                error = str(e)
            with self._lock:
                self._running = False
                self._last_result = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "added": added,
                    "limit": limit,
                    "error": error,
                }
            if callback:
                try:
                    callback()
                except Exception as e:
                    print(f"[调度器] 抓取完成回调失败: {e}")

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()
        print(f"[调度器] 抓取任务已启动（每频道 {limit} 条）")
        return True

    def start_auto(self, callback=None):
        """
        启动后台定时抓取循环（daemon 线程，随 Web 退出而结束）。
        首次抓取在 Web 启动 8 秒后执行，避免影响启动速度。
        """
        def _loop():
            time.sleep(8)  # 等 Web 完全启动
            while True:
                try:
                    self.start(limit=config.AUTO_CRAWL_LIMIT, callback=callback)
                except Exception as e:
                    print(f"[调度器] 定时抓取异常: {e}")
                time.sleep(config.AUTO_CRAWL_INTERVAL)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        print(f"[调度器] 自动抓取已开启，间隔 {config.AUTO_CRAWL_INTERVAL} 秒")


# 全局单例
crawl_manager = CrawlManager()
