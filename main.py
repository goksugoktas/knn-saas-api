from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import io
from typing import List, Dict, Any

# --- ORİJİNAL STANDART AÇIKLAMA METNİ ---
description = """
**KNN Klasik Sınıflandırma SaaS API** etiketli veri setlerinizi yükleyerek K-En Yakın Komşu algoritmasıyla modeller kurmanızı sağlar. 🛠️

## İş Akışı ve Kullanım Talimatı
1. **Veri Yükleme (`/dataset/upload`):** CSV formatında, en son kolonu hedef etiket (target) olan veri setinizi yükleyin.
2. **Yapılandırma (`/model/configure`):** K değerini, mesafe metriğini ve ağırlık tipini belirleyerek modeli eğitin.
3. **Tahmin (`/predict`):** Modelin eğitildiği özellik sırasına uygun olarak yeni veri gönderin, sınıfı ve en yakın komşuları görün.
"""

# CRITICAL BULUT FIX: Tasarım yok, sadece 404 hatasını çözen yönlendirme var
app = FastAPI(
    title="KNN Classifier SaaS System",
    description=description,
    version="1.0.0",
    docs_url=None, 
    redoc_url=None,
    openapi_url="/openapi.json"
)

# 1. MERKEZİ VERİ DEPOLAMA (STATEFUL STORAGE)
storage = {
    "df": None, 
    "model": None, 
    "feature_names": None, 
    "target_name": None
}

# 2. VERİ MODELLERİ (PYDANTIC)
class ConfigParams(BaseModel):
    k: int = Field(5, description="En yakın komşu sayısı (K değeri)", gt=0)
    metric: str = Field("euclidean", description="Mesafe hesaplama metriği (euclidean, manhattan, minkowski)")
    weights: str = Field("uniform", description="Ağırlıklandırma türü (uniform, distance)")

class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Özellik isimleri ve değerlerini içeren JSON obje. Örn: {'feature1': 2.5, 'feature2': 4.1}")

# --- STANDART/ORİJİNAL SWAGGER UI ARAYÜZÜ ---
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    # CSS kodları tamamen temizlendi, orijinal Swagger UI çağrılıyor
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title,
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# 3. ENDPOINTLER 

@app.get("/", tags=["Genel Kontrol"])
def home():
    return {
        "durum": "Aktif",
        "mesaj": "KNN SaaS API başarıyla çalışıyor. Test etmek için lütfen /docs adresine gidin."
    }

# ENDPOINT 1: Veri Kümesi Yükleme
@app.post("/dataset/upload", tags=["Veri Yönetimi"])
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir CSV dosyası yükleyin.")
    
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Veri seti en az bir özellik ve bir etiket kolonu içermelidir.")
        
        storage["df"] = df
        storage["feature_names"] = df.columns[:-1].tolist()
        storage["target_name"] = df.columns[-1]
        storage["model"] = None 
        
        return {
            "mesaj": "Veri seti başarıyla yüklendi ve doğrulandı.",
            "satir_sayisi": df.shape[0],
            "sutun_sayisi": df.shape[1],
            "ozellikler": storage["feature_names"],
            "hedef_etiket": storage["target_name"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veri işleme hatası: {str(e)}")

# ENDPOINT 2: Model Yapılandırma ve Eğitme
@app.post("/model/configure", tags=["Model İşlemleri"])
async def configure_model(params: ConfigParams):
    if storage["df"] is None:
        raise HTTPException(status_code=400, detail="Hata: Model eğitilmeden önce /dataset/upload adresinden veri yüklenmelidir.")
    
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
            "mesaj": "KNN Modeli başarıyla yapılandırıldı ve eğitildi.",
            "parametreler": {
                "k": k_value,
                "metric": params.metric,
                "weights": params.weights
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model eğitim hatası: {str(e)}")

# ENDPOINT 3: Tahminleme ve Komşuları Döndürme
@app.post("/predict", tags=["Tahminleme (Predict)"])
async def predict(input_data: PredictionInput):
    if storage["model"] is None:
        raise HTTPException(status_code=400, detail="Hata: Tahmin yapabilmek için önce model eğitilmelidir.")
    
    try:
        input_dict = input_data.features
        
        missing_cols = [col for col in storage["feature_names"] if col not in input_dict]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Eksik özellik verisi gönderildi. Gerekli kolonlar: {missing_cols}")
        
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
            "tahmin_edilen_sinif": int(prediction),
            "en_yakin_komsu_sayisi": len(neighbors_list),
            "en_yakin_komsular": neighbors_list
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Sistem Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tahmin işlem esnasında bir hata meydana geldi: {str(e)}")
