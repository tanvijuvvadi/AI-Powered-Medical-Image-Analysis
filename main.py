"""Command-line entry point.

  python main.py explore            # dataset stats + sample images
  python main.py train              # train MobileNetV2 (add --baseline for simple CNN)
  python main.py evaluate           # test metrics + confusion matrix + ROC
  python main.py predict IMG.jpeg   # prediction + Grad-CAM figure
  python main.py serve              # Flask API on http://127.0.0.1:5000
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("explore")
    t = sub.add_parser("train"); t.add_argument("--baseline", action="store_true")
    sub.add_parser("evaluate")
    pr = sub.add_parser("predict"); pr.add_argument("image")
    sub.add_parser("serve")
    a = p.parse_args()

    if a.cmd == "explore":
        from src.data_loader import count_images
        from src.visualize import plot_class_distribution, plot_preprocessing_steps, plot_sample_images
        from src.config import DATA_DIR
        counts = {s: count_images(s) for s in ("train", "val", "test")}
        print(counts)
        plot_class_distribution(counts)
        plot_sample_images()
        first = next((DATA_DIR / "train" / "PNEUMONIA").glob("*.jp*g"))
        plot_preprocessing_steps(first)
    elif a.cmd == "train":
        from src.train import train
        train(baseline=a.baseline)
    elif a.cmd == "evaluate":
        from src.evaluate import evaluate
        evaluate()
    elif a.cmd == "predict":
        from src.predict import predict_file
        predict_file(a.image)
    elif a.cmd == "serve":
        from src.api import run
        run()


if __name__ == "__main__":
    main()
