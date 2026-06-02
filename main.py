from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import io
from typing import List, Dict, Any

description = """
**Dinamik KNN Endüstriyel SaaS API** - Metinsel kolonları (Product ID, Type vb.) otomatik olarak temizler ve sadece sayısal verilerle modeli eğitir. 
"""

app = FastAPI(
    title="KNN Classifier SaaS System",
    description=description,
    version="1.1.0",
    docs_url=None, 
    redoc_url=None,
    openapi_url="/openapi.json"
)

storage = {
    "df": None, 
    "model": None, 
    "feature_names": None, 
    "target_name": None
}

class ConfigParams(BaseModel):
    k: int = Field(5, description="En yakın komşu sayısı (K değeri)", gt=0)
    metric: str = Field("euclidean", description="Mesafe hesaplama metriği (euclidean, manhattan, minkowski)")
    weights: str = Field("uniform", description="Ağırlıklandırma türü (uniform, distance)")

class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Sadece sayısal özellik değerlerini içeren JSON obje.")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title,
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/", tags=["Genel Kontrol"])
def home():
    return {"durum": "Aktif", "mesaj": "Sistem aktif. /docs adresine gidin."}

# ENDPOINT 1: Veri Kümesi Yükleme
@app.post("/dataset/upload", tags=["Veri Yönetimi"])
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir CSV dosyası yükleyin.")
    
    content = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(content))
        
        target_col = raw_df.columns[-1]
        
        feature_candidates = raw_df.columns[:-1].tolist()
        clean_features = []
        
        for col in feature_candidates:
            if pd.api.types.is_numeric_dtype(raw_df[col]):
                clean_features.append(col)
        
        df = raw_df[clean_features + [target_col]].copy()
        
        storage["df"] = df
        storage["feature_names"] = clean_features
        storage["target_name"] = target_col
        storage["model"] = None 
        
        return {
            "mesaj": "Veri seti yüklendi. Sayısal olmayan kolonlar (Product ID, Type vb.) otomatik olarak temizlendi!",
            "satir_sayisi": df.shape[0],
            "aktif_sutun_sayisi": df.shape[1],
            "isleme_alinan_ozellikler": storage["feature_names"],
            "hedef_etiket": storage["target_name"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veri işleme hatası: {str(e)}")

# ENDPOINT 2: Model Yapılandırma
@app.post("/model/configure", tags=["Model İşlemleri"])
async def configure_model(params: ConfigParams):
    if storage["df"] is None:
        raise HTTPException(status_code=400, detail="Hata: Önce veri yüklenmelidir.")
    
    if params.metric.lower() not in ["euclidean", "manhattan", "minkowski"]:
        raise HTTPException(status_code=400, detail="Desteklenen metrikler: 'euclidean', 'manhattan', 'minkowski'")

    try:
        X = storage["df"][storage["feature_names"]]
        y = storage["df"][storage["target_name"]]
        k_value = min(params.k, len(X))
        
        knn = KNeighborsClassifier(n_neighbors=k_value, metric=params.metric.lower(), weights=params.weights)
        knn.fit(X, y)
        storage["model"] = knn
        
        return {
            "mesaj": "KNN Modeli sadece sayısal kolonlarla başarıyla eğitildi.",
            "parametreler": {"k": k_value, "metric": params.metric, "weights": params.weights}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model eğitim hatası: {str(e)}")

# ENDPOINT 3: Tahminleme (GİRİNTİLER VE RETURN BLOKLARI DÜZELTİLDİ)
@app.post("/predict", tags=["Tahminleme (Predict)"])
async def predict(input_data: PredictionInput):
    if storage["model"] is None:
        raise HTTPException(status_code=400, detail="Hata: Önce model eğitilmelidir.")
    
    try:
        input_dict = input_data.features
        missing_cols = [col for col in storage["feature_names"] if col not in input_dict]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Eksik veri. Gerekli kolonlar: {missing_cols}")
        
        input_df = pd.DataFrame([input_dict])[storage["feature_names"]]
        prediction = storage["model"].predict(input_df)[0]
        distances, indices = storage["model"].kneighbors(input_df)
        
        neighbors_list = []
        for idx, dist in zip(indices[0], distances[0]):
            neighbor_row = storage["df"].iloc[idx].to_dict()
            neighbors_list.append({
                "orijinal_veri_indeksi": int(idx),
                "mesafe": float(dist),
                "detaylar": neighbor_row
            })
            
        return {
            "tahmin_edilen_sinif": str(prediction),
            "en_yakin_komsu_sayisi": len(neighbors_list),
            "en_yakin_komsular": neighbors_list
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")
