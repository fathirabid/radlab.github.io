import joblib
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load model dan scaler
model = joblib.load("python/models/esi_model.pkl")
features_count = 5 # Berdasarkan 'features' di train_model.py

# Tentukan tipe input (5 fitur sesuai train_model.py)
initial_type = [('float_input', FloatTensorType([None, features_count]))]

# Konversi
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Simpan
with open("model_esi.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())