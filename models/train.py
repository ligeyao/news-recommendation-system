# -*- coding: utf-8 -*-
"""
新闻文本分类模型训练
====================
算法对比：朴素贝叶斯、线性 SVM、逻辑回归（TF-IDF 特征）
流程：加载数据 -> 分词 -> TF-IDF -> 训练/测试划分 -> 训练对比 -> 保存最优模型

运行方式：
    python -m models.train                 # 默认从 data/news.db 读数据
    python -m models.train --data data/news.csv   # 指定 CSV 数据
    python -m models.train --model svm     # 只训练某一种（nb / svm / lr）
    python -m models.train --plot          # 额外保存混淆矩阵图
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from models.preprocess import (  # noqa: E402
    build_tfidf,
    load_news,
    load_news_from_csv,
    load_stopwords,
    save_pickle,
    tokenize,
)

# 参与对比的模型
MODEL_FACTORIES = {
    "nb": ("朴素贝叶斯", lambda: MultinomialNB(alpha=0.5)),
    "svm": ("线性SVM", lambda: LinearSVC(C=1.0, max_iter=3000, random_state=config.RANDOM_STATE)),
    "lr": ("逻辑回归", lambda: LogisticRegression(max_iter=3000, random_state=config.RANDOM_STATE)),
}


def prepare_data(df):
    """
    数据预处理：
    1. 丢弃类别样本数过少的类
    2. 标题 + 正文拼接后分词
    返回 (texts_tokenized, labels, label_names)
    """
    df = df.dropna(subset=["content", "category"]).copy()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"] != ""]

    # 过滤样本过少的类别
    counts = df["category"].value_counts()
    keep = counts[counts >= config.MIN_SAMPLES_PER_CLASS].index.tolist()
    df = df[df["category"].isin(keep)]
    if len(keep) < 2:
        raise ValueError(f"有效类别数少于 2（当前 {len(keep)}），请先爬取更多数据")

    # 标题和正文拼接，给标题更高权重（重复一次）
    stopwords = load_stopwords()
    texts = (df["title"].fillna("") + " " + df["title"].fillna("") + " " + df["content"].fillna(""))
    tokenized = texts.map(lambda t: tokenize(t, stopwords))

    labels = df["category"].tolist()
    label_names = sorted(set(labels))
    print(f"有效样本: {len(df)} 条，类别: {label_names}")
    print("各类样本数:")
    for name in label_names:
        print(f"  {name}: {labels.count(name)}")
    return tokenized, labels, label_names


def train_models(X_train, y_train, X_test, y_test, label_names, only=None):
    """训练多个模型并对比评估，返回 (结果列表, 训练好的模型字典)。"""
    results = []
    trained = {}

    for key, (name, factory) in MODEL_FACTORIES.items():
        if only and key != only:
            continue
        print(f"\n训练模型: {name} ...")
        t0 = time.time()
        model = factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        results.append({"key": key, "name": name, "model": model,
                        "accuracy": acc, "y_pred": y_pred})
        trained[key] = model
        print(f"  测试集准确率: {acc:.4f}（耗时 {elapsed:.2f}s）")
        print("  分类报告:")
        print(classification_report(y_test, y_pred, labels=label_names, zero_division=0))
    return results, trained


def save_best(results, vectorizer, label_names):
    """选择准确率最高的模型保存（连同向量器和标签）。"""
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n最优模型: {best['name']}，准确率 {best['accuracy']:.4f}")
    save_pickle(best["model"], config.MODEL_PATH)
    save_pickle(vectorizer, config.VECTORIZER_PATH)
    save_pickle(label_names, config.LABELS_PATH)
    print(f"模型已保存: {config.MODEL_PATH}")
    print(f"向量器已保存: {config.VECTORIZER_PATH}")
    print(f"标签已保存: {config.LABELS_PATH}")
    return best


def save_report(results, y_test, label_names):
    """把所有模型的分类报告汇总写入文本文件。"""
    lines = [f"新闻分类模型对比报告（生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}）",
             "=" * 60]
    for r in results:
        lines.append(f"\n模型: {r['name']}  准确率: {r['accuracy']:.4f}")
        lines.append(classification_report(y_test, r["y_pred"], labels=label_names, zero_division=0))
    report = "\n".join(lines)
    Path(config.REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n评估报告已保存: {config.REPORT_PATH}")


def plot_confusion_matrices(results, y_test, label_names):
    """为每个模型绘制混淆矩阵并保存 PNG。"""
    import matplotlib
    matplotlib.use("Agg")  # 无界面环境也能保存图片
    import matplotlib.pyplot as plt

    # 中文字体设置（Windows 用 SimHei）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"], labels=label_names)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['name']} 混淆矩阵", fontsize=12)
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, fontsize=9)
        ax.set_yticklabels(label_names, fontsize=9)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = Path(config.MODEL_DIR) / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"混淆矩阵图已保存: {out}")


def main():
    parser = argparse.ArgumentParser(description="新闻文本分类模型训练")
    parser.add_argument("--data", type=str, default="", help="CSV 数据路径（默认自动选择 DB/CSV）")
    parser.add_argument("--model", type=str, default="", choices=["nb", "svm", "lr", ""],
                        help="只训练某一种模型，默认全部对比")
    parser.add_argument("--plot", action="store_true", help="保存混淆矩阵图")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE, help="测试集比例")
    args = parser.parse_args()

    # 1. 加载数据
    print("=" * 60)
    print("加载数据...")
    if args.data:
        df = load_news_from_csv(args.data)
    else:
        df = load_news()
    print(f"原始数据: {len(df)} 条")

    # 2. 预处理：分词
    print("\n数据预处理（分词）...")
    tokenized, labels, label_names = prepare_data(df)

    # 3. TF-IDF 特征
    print("\n提取 TF-IDF 特征...")
    vectorizer, X = build_tfidf(tokenized)

    # 4. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=args.test_size,
        random_state=config.RANDOM_STATE, stratify=labels,
    )
    print(f"训练集: {X_train.shape[0]} 条，测试集: {X_test.shape[0]} 条，特征维度: {X_train.shape[1]}")

    # 5. 训练对比
    results, _ = train_models(X_train, y_train, X_test, y_test, label_names, only=args.model or None)

    # 6. 保存最优模型 + 报告
    best = save_best(results, vectorizer, label_names)
    save_report(results, y_test, label_names)

    if args.plot:
        plot_confusion_matrices(results, y_test, label_names)

    print("\n训练完成！Web 系统可直接加载保存的模型进行预测。")
    return best


if __name__ == "__main__":
    main()
