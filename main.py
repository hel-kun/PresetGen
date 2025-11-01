import argparse
from datetime import datetime
import logging
import torch
from dataset.dataset import Synth1Dataset
from model.model import PresetGenModel
from model.trainer import Trainer
from model.loss import ParamsLoss
from config import DEVICE

def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    dataset = Synth1Dataset(logger=logger)
    model = PresetGenModel(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    if args.eval_only or args.resume_from_checkpoint:
        date = args.resume_from_checkpoint.split('/')[-2]
    else: 
        date = datetime.now().strftime("%Y%m%d_%H%M%S")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        checkpoint_path=f'checkpoints/{date}',
        log_interval=args.log_interval,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        criterion=ParamsLoss(),
        early_stopping_patience=args.es_patience,
        logger=logger
    )
    if not args.eval_only:
        trainer.train(
            num_epochs=args.epochs,
            resume_from_checkpoint=args.resume_from_checkpoint
        )

    trainer.evaluate(trainer.test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PresetGen")

    # データセットとモデルのパラメータ
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of transformer attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # トレーニングのパラメータ
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--es-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-interval", type=int, default=10, help="Model save interval (in epochs)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (in batches)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")

    main(parser.parse_args())
