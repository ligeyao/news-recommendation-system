# 基于机器学习的新闻推荐系统

> 课程设计 / 毕业设计项目：爬取网易新闻 → 数据清洗入库 → 文本分类模型训练 → 基于内容的个性化推荐 → Web 可视化展示。

## ✨ 功能特性

- **实时新闻抓取**：网易新闻 8 大频道自动抓取，支持定时自动更新 + 网页一键手动刷新
- **数据持久化**：SQLite 存储，自动去重，含数据清洗模块
- **新闻文本分类**：jieba 分词 + TF-IDF 特征，朴素贝叶斯 / 线性 SVM / 逻辑回归三种算法对比，自动保存最优模型
- **个性化推荐**：基于内容的推荐算法（TF-IDF + 余弦相似度 + 类别多样性），支持基于浏览历史推荐、单条相似新闻推荐、关键词搜索推荐
- **Web 可视化**：Flask + Jinja2，新闻列表（分页、分类统计）、新闻详情（相关推荐）、个性化推荐页、搜索推荐、模型评估页

## 🛠 技术栈

| 模块 | 技术 |
|---|---|
| 爬虫 | requests、BeautifulSoup4、lxml |
| 存储 | SQLite（sqlite3） |
| 文本处理 | jieba 分词、停用词过滤 |
| 特征工程 | TF-IDF（scikit-learn） |
| 分类算法 | MultinomialNB、LinearSVC、LogisticRegression |
| 推荐算法 | 余弦相似度 + 兴趣画像 + 类别多样性 |
| Web | Flask、Jinja2、原生 HTML/CSS/JS |

## 📁 项目结构

```
News/
├── config.py                 # 全局配置（路径、频道、爬虫/模型/抓取参数）
├── requirements.txt          # Python 依赖清单
├── crawler/
│   ├── spider.py             # 网易新闻爬虫（列表接口 + HTML兜底 + 正文提取 + 入库）
│   ├── clean.py              # 数据清洗（去HTML、去重、过滤、导出CSV）
│   └── scheduler.py          # 抓取调度器（定时自动抓取 + 手动触发）
├── models/
│   ├── preprocess.py         # 分词、停用词、TF-IDF、数据加载
│   ├── train.py              # 分类模型训练与算法对比
│   └── saved/                # 模型保存目录（训练后生成，已被 git 忽略）
├── recommend/
│   └── recommender.py        # 基于内容的推荐器
├── web/
│   ├── app.py                # Flask 主程序（所有路由）
│   ├── templates/            # 页面模板（base/index/detail/recommend/evaluate/404）
│   └── static/               # 静态资源
└── data/
    ├── news.db               # SQLite 新闻库（已含测试数据）
    └── stopwords.txt         # 中文停用词表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 爬取新闻（仓库已带测试数据，可跳过）

```bash
python -m crawler.spider            # 爬取全部 8 个频道
python -m crawler.spider --cat 科技  # 只爬某个频道
python -m crawler.spider --limit 50  # 每个频道最多 50 条
python -m crawler.clean --export     # 清洗数据并导出 CSV
```

### 3. 训练分类模型（可选，用于评估页展示）

```bash
python -m models.train --plot        # 训练对比 3 种算法，保存最优模型和混淆矩阵
```

### 4. 启动 Web 系统

```bash
python -m web.app
```

浏览器访问：**http://127.0.0.1:5000**

## 🖥 页面说明

| 路由 | 页面 | 说明 |
|---|---|---|
| `/` | 最新新闻 | 分页列表、分类统计栏 |
| `/category/<name>` | 分类新闻 | 按频道浏览 |
| `/news/<id>` | 新闻详情 | 全文 + 相似推荐，自动记录浏览历史 |
| `/recommend` | 我的推荐 | 基于浏览历史的个性化推荐，可清空历史 |
| `/search?q=关键词` | 搜索推荐 | 基于文本相似度的新闻检索 |
| `/evaluate` | 模型评估 | 分类报告 + 混淆矩阵 |
| `/crawl` | 立即抓取 | POST 触发后台抓取（导航栏按钮） |
| `/crawl_status` | 抓取状态 | JSON 接口，前端轮询显示 |

## 🧠 推荐算法说明

**基于内容的推荐（Content-Based Recommendation）**：

1. 每条新闻用 TF-IDF 向量表示（标题权重加倍：标题重复两次 + 正文）
2. 用户浏览历史中的所有新闻向量取平均 → 得到用户**兴趣向量**
3. 兴趣向量与全库新闻计算**余弦相似度**
4. 排除已读新闻，做**类别多样性**挑选后按相似度降序返回 Top-N
5. 无浏览历史时冷启动兜底：返回最新新闻

**分类类别**：财经、体育、娱乐、汽车、科技、社会、军事、女性

> 注：网易"游戏/旅游"频道公开接口已关闭（404），故用数据质量更好的"军事/女性"频道替代，分类可在 `config.py` 中调整。

## ⚙️ 主要配置（config.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `AUTO_CRAWL_ENABLED` | `True` | Web 启动后是否自动抓取 |
| `AUTO_CRAWL_INTERVAL` | `3600` | 自动抓取间隔（秒） |
| `AUTO_CRAWL_LIMIT` | `30` | 自动抓取每频道条数 |
| `MANUAL_CRAWL_LIMIT` | `50` | 手动抓取每频道条数 |
| `CRAWL_DELAY` | `0.5` | 请求间隔（秒），礼貌爬取 |
| `MAX_FEATURES` | `20000` | TF-IDF 最大特征数 |
| `MIN_SAMPLES_PER_CLASS` | `10` | 训练时类别最小样本数 |
| `WEB_PORT` | `5000` | Web 服务端口 |

## ❓ 常见问题

**1. 爬虫抓不到数据怎么办？**

网易接口可能改版失效。课题允许使用公开数据集：把数据整理成 CSV（列：`title, content, category`）放到 `data/news.csv`，训练脚本会自动加载。

**2. 推荐效果不明显？**

数据库新闻太少（如只有几十条），建议运行 `python -m crawler.spider` 爬全量数据（约 800 条），推荐质量会明显提升。

**3. 端口被占用？**

修改 `config.py` 中的 `WEB_PORT`，或先结束占用 5000 端口的进程。

**4. 命令行中文乱码？**

Windows 下运行时加环境变量：`PYTHONIOENCODING=utf-8 python -m web.app`

## 📄 许可

本项目仅用于课程学习与毕业设计。
