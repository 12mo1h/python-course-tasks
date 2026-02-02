#Mohamed ashraf salah
import matplotlib.pyplot as plt

# بيانات الأيام ودرجات الحرارة
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
temperatures = [22, 24, 23, 25, 26, 27, 24]

# إنشاء الرسم البياني الخطي
plt.plot(days, temperatures, marker='o', linestyle='-', color='blue')

# إضافة العنوان والتسميات
plt.title("Weekly Temperature Variation")
plt.xlabel("Days of the Week")
plt.ylabel("Temperature (°C)")

# عرض الشبكة لتحسين القراءة
plt.grid(True)

# عرض الرسم
plt.show()
