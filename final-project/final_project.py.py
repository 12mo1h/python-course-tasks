#mohamed ashrtaf salah
# final task
# مكتبات
import numpy as np              # مكتبة العمليات الحسابية
import pandas as pd             # مكتبة معالجة البيانات
import matplotlib.pyplot as plt # مكتبة الرسم البياني

data = pd.read_csv("./python course/student_score.csv")
print(data.head())
print("\nMissing Values:")
print(data.isnull().sum())


data.fillna(data.mean(), inplace=True)

plt.scatter(data["Hours"], data["Scores"], color="blue")
plt.title("Study Hours vs Student Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Student Score")
plt.grid(True)
plt.show()

#  تحديد المتغيرات
#DataFrame
X = data[["Hours"]]   # المتغير المستقل (المدخل)
y = data["Scores"]    # المتغير التابع (المخرج)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=42     
)


from sklearn.linear_model import LinearRegression

model = LinearRegression()  # إنشاء النموذج
model.fit(X_train, y_train) # تدريب النموذج

#  التنبؤ باستخدام بيانات الاختبار
y_pred = model.predict(X_test)

#  تقييم النموذج
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)  # حساب MSE
r2 = r2_score(y_test, y_pred)              # حساب R²

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-Squared (R²):", r2)


plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Linear Regression - Student Score Prediction")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
