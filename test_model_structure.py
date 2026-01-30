import pickle
from pathlib import Path

# Test loading a model file to examine its structure
model_path = Path("ml_models") / "ensemble_BTCUSDT_15_15m.pkl"

try:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    print("Model data keys:", list(model_data.keys()))
    
    # Check if 'model' key exists
    if "model" in model_data:
        print("'model' key exists")
        print("Model type:", type(model_data["model"]))
    else:
        print("'model' key does NOT exist")
        
    # Check other keys
    for key in model_data.keys():
        print(f"{key}: {type(model_data[key])}")
        if key == "metadata":
            print(f"  Metadata keys: {list(model_data[key].keys()) if hasattr(model_data[key], 'keys') else 'N/A'}")
            
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()