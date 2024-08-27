# بناء واجهة برمجية (API)

# للسماح للأطباء والممرضين بإدخال بيانات المرضى واسترجاع التشخيصات  

# استدعاء المكتبات الازمه
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# 1. إنشاء التطبيق باستخدام Flask
app = Flask(__name__)

# 2. تحميل النموذج المدرب
model = load_model('project_1.keras')

# 3. ترميز الأعمدة الفئوية (مرة واحدة للاستخدام المتعدد)
label_encoders = {
    'sex': LabelEncoder(),
    'cp': LabelEncoder(),
    'restecg': LabelEncoder(),
    'exang': LabelEncoder()
}

# تجهيز القيم الفئوية الممكنة
label_encoders['sex'].fit(['male', 'female'])
label_encoders['cp'].fit(['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
label_encoders['restecg'].fit(['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
label_encoders['exang'].fit(['yes', 'no'])

scaler = StandardScaler()

#  نقطة النهاية للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():



    # الحصول على بيانات المريض من الطلب
    data = request.json
    
    # تحويل بيانات المريض إلى DataFrame
    df = pd.DataFrame([data])

    # ترميز القيم الفئوية
    for column in label_encoders.keys():
        df[column] = label_encoders[column].transform(df[column])

    # تقيسس البيانات
    df_scaled = scaler.fit_transform(df)

    # التنبؤ باستخدام النموذج
    prediction = model.predict(df_scaled)
    result = (prediction[0] > 0.5).astype(int)

    # عرض النتيجة
    diagnosis = "من المحتمل أن يكون المريض مصاباً بمرض القلب" if result == 1 else "من غير المحتمل أن يكون المريض مصاباً بمرض القلب"
    return jsonify({"diagnosis": diagnosis, "probability": float(prediction[0])})

#  تشغيل التطبيق
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
