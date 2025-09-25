import os

# 避免 OpenMP 运行时冲突（libiomp5md.dll 重复初始化）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from training import SelfPlayTrainer

# 创建训练器
trainer = SelfPlayTrainer(
    model_path=None,           # 预训练模型路径（可选）
    device="cpu",             # 强制使用CPU
    learning_rate=0.001,      # 学习率
    batch_size=512            # 批大小
)

# 开始训练
trainer.train(
    num_iterations=10,          # 进一步减少迭代次数
    games_per_iteration=30,     # 增加每次迭代的游戏数量以提高数据质量
    epochs_per_iteration=3     # 减少训练轮数避免过拟合
)