import pickle
import sys
from pathlib import Path

def examine_pickle_file(filepath):
    """Examine a pickle file without loading the full content."""
    try:
        with open(filepath, 'rb') as f:
            # Load the pickle file
            data = pickle.load(f)
            
        print(f"File: {filepath}")
        print(f"Type of loaded data: {type(data)}")
        
        if isinstance(data, dict):
            print("Keys in the dictionary:")
            for key in data.keys():
                print(f"  {key}: {type(data[key])}")
                
                # If it's a nested dictionary, show its keys
                if isinstance(data[key], dict):
                    print(f"    Nested keys: {list(data[key].keys())}")
                elif hasattr(data[key], '__dict__'):
                    print(f"    Object attributes: {list(data[key].__dict__.keys())}")
        else:
            print("Data is not a dictionary")
            
        # Check specifically for the 'model' key
        if isinstance(data, dict) and 'model' in data:
            print("\nFound 'model' key!")
            print(f"Model type: {type(data['model'])}")
        else:
            print("\n'model' key NOT found!")
            
        return data
        
    except Exception as e:
        print(f"Error examining file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "ml_models/ensemble_BTCUSDT_15_15m.pkl"
    
    print(f"Examining: {filepath}")
    data = examine_pickle_file(filepath)