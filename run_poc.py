import os
import sys
import subprocess
import time


def print_banner():
    print("=" * 75)
    print("      YELLOWSENSE — UNCERTAINTY-AWARE MOBILITY INTELLIGENCE")
    print("       COMET 5G SPRING BOARD 6 | PROBLEM STATEMENT #6")
    print("=" * 75)


def check_prerequisites():
    print("\n[1/4] Checking environment prerequisites...")
    required_packages = ["numpy", "pandas", "scikit-learn", "xgboost", "fastapi", "uvicorn", "streamlit", "plotly"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[!] Missing packages: {missing}")
        print("    Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("      All required packages installed.")

    # 2. Check Dataset
    print("\n[2/4] Verifying synthetic mobility dataset...")
    data_path = os.path.join("data", "synthetic", "synthetic_mobility_dataset.parquet")
    if not os.path.exists(data_path):
        print("      Dataset not found. Generating synthetic multi-cell trajectories...")
        subprocess.check_call([sys.executable, "scripts/generate_data.py"])
    else:
        print(f"      Dataset verified: {data_path}")

    # 3. Check Models
    print("\n[3/4] Verifying trained & calibrated ML models...")
    next_cell_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    weakness_path = os.path.join("artifacts", "models", "weakness_model.joblib")
    if not os.path.exists(next_cell_path) or not os.path.exists(weakness_path):
        print("      Model artifacts not found. Running training & calibration...")
        subprocess.check_call([sys.executable, "scripts/train.py"])
    else:
        print("      Model artifacts verified in artifacts/models/")

    # 4. Check Evaluation Reports
    print("\n[4/4] Verifying evaluation reports & figures...")
    report_path = os.path.join("artifacts", "reports", "evaluation_report.md")
    if not os.path.exists(report_path):
        subprocess.check_call([sys.executable, "scripts/evaluate.py"])
    else:
        print("      Evaluation artifacts verified in artifacts/reports/")


def main():
    print_banner()
    check_prerequisites()

    print("\n" + "=" * 75)
    print("SELECT EXECUTION MODE:")
    print("=" * 75)
    print("1. Launch Interactive Streamlit Console (Recommended)")
    print("2. Run Live CLI Highway Route Deviation Demo")
    print("3. Start FastAPI Microservice (http://127.0.0.1:8000)")
    print("4. Re-run Full Evaluation & Benchmarking")
    print("5. Run Complete Pytest Automated Test Suite")
    print("6. Exit")
    print("-" * 75)

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        try:
            choice = input("Enter choice [1-6] (Default: 1): ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            choice = "1"

    if choice == "1":
        print("\nStarting Streamlit Console on http://localhost:8501 ...")
        subprocess.call([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
    elif choice == "2":
        subprocess.call([sys.executable, "scripts/run_demo.py"])
    elif choice == "3":
        print("\nStarting FastAPI Server on http://127.0.0.1:8000 ...")
        subprocess.call([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    elif choice == "4":
        subprocess.call([sys.executable, "scripts/evaluate.py"])
    elif choice == "5":
        subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"])
    else:
        print("\nExiting. All components ready.")


if __name__ == "__main__":
    main()
