import os


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def check_import(name, import_name=None):
    import importlib

    module_name = import_name or name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
        return module
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return None


def main():
    print("Checking basic packages...")
    check_import("numpy")
    check_import("pandas")
    check_import("scipy")
    check_import("scikit-learn", "sklearn")
    check_import("rdkit")
    check_import("networkx")
    check_import("tqdm")
    check_import("tensorboardX")

    print("\nChecking PyTorch...")
    torch = check_import("torch")
    if torch is not None:
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"is CU128: {torch.version.cuda == '12.8'}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(2, 2).to(device)
        y = x @ x
        print(f"actual torch device: {y.device}")
        print(f"using CUDA: {y.is_cuda}")
        if y.is_cuda:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    print("\nChecking PyTorch Geometric...")
    check_import("torch_geometric")
    check_import("torch_scatter")
    check_import("pyg_lib")
    check_import("torch_sparse")

    print("\nDone.")


if __name__ == "__main__":
    main()
