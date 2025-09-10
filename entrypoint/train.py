import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipelines.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    try:
        pipeline = TrainingPipeline(args.config)
        results = pipeline.run()
        print(f"Training completed! Accuracy: {results['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()