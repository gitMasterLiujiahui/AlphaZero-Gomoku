"""
Enhanced Training Launcher
增强训练启动器

提供多种训练模式的统一入口：
- 基础训练
- 并行训练
- 课程学习训练
- 对抗性训练
"""

import os
import argparse
import logging
from typing import Optional

# 避免 OpenMP 运行时冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from training import SelfPlayTrainer
from parallel_training import ParallelSelfPlayTrainer
from curriculum_training import CurriculumAdversarialTrainer


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def basic_training(model_path: Optional[str] = None, 
                  learning_rate: float = 0.001,
                  batch_size: int = 512,
                  num_iterations: int = 10,
                  games_per_iteration: int = 30,
                  epochs_per_iteration: int = 3):
    """基础训练模式"""
    print("Starting Basic Training Mode...")
    
    trainer = SelfPlayTrainer(
        model_path=model_path,
        device="cpu",
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    trainer.train(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        epochs_per_iteration=epochs_per_iteration
    )


def parallel_training(model_path: Optional[str] = None,
                     learning_rate: float = 0.001,
                     batch_size: int = 256,
                     num_workers: int = 2,
                     num_iterations: int = 10,
                     games_per_iteration: int = 30,
                     epochs_per_iteration: int = 3):
    """并行训练模式"""
    print("Starting Parallel Training Mode...")
    
    trainer = ParallelSelfPlayTrainer(
        model_path=model_path,
        device="cpu",
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    trainer.train(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        epochs_per_iteration=epochs_per_iteration
    )


def curriculum_training(model_path: Optional[str] = None,
                       learning_rate: float = 0.001,
                       batch_size: int = 256,
                       num_workers: int = 2,
                       num_iterations: int = 10,
                       games_per_iteration: int = 30,
                       epochs_per_iteration: int = 3,
                       curriculum_ratio: float = 0.6):
    """课程学习和对抗性训练模式"""
    print("Starting Curriculum-Adversarial Training Mode...")
    
    trainer = CurriculumAdversarialTrainer(
        model_path=model_path,
        device="cpu",
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    trainer.train(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        epochs_per_iteration=epochs_per_iteration,
        curriculum_ratio=curriculum_ratio
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced AlphaZero Training Framework")
    
    # 训练模式
    parser.add_argument("--mode", type=str, default="basic",
                       choices=["basic", "parallel", "curriculum"],
                       help="Training mode: basic, parallel, or curriculum")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    
    # 训练参数
    parser.add_argument("--num_iterations", type=int, default=5,
                       help="Number of training iterations")
    parser.add_argument("--games_per_iteration", type=int, default=20,
                       help="Number of games per iteration")
    parser.add_argument("--epochs_per_iteration", type=int, default=3,
                       help="Number of training epochs per iteration")
    
    # 并行训练参数
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of parallel workers")
    
    # 课程学习参数
    parser.add_argument("--curriculum_ratio", type=float, default=0.6,
                       help="Ratio of curriculum learning games")
    
    # 日志参数
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    print("=" * 60)
    print("Enhanced AlphaZero-Gomoku Training Framework")
    print("=" * 60)
    print(f"Training Mode: {args.mode}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Games per Iteration: {args.games_per_iteration}")
    print(f"Epochs per Iteration: {args.epochs_per_iteration}")
    print("=" * 60)
    
    try:
        if args.mode == "basic":
            basic_training(
                model_path=args.model_path,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_iterations=args.num_iterations,
                games_per_iteration=args.games_per_iteration,
                epochs_per_iteration=args.epochs_per_iteration
            )
        elif args.mode == "parallel":
            parallel_training(
                model_path=args.model_path,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_iterations=args.num_iterations,
                games_per_iteration=args.games_per_iteration,
                epochs_per_iteration=args.epochs_per_iteration
            )
        elif args.mode == "curriculum":
            curriculum_training(
                model_path=args.model_path,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_iterations=args.num_iterations,
                games_per_iteration=args.games_per_iteration,
                epochs_per_iteration=args.epochs_per_iteration,
                curriculum_ratio=args.curriculum_ratio
            )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        logging.error(f"Training error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
