#mohamed ashraf salah

# 1. استيراد مكتبة numpy
import numpy as np


# 2. إنشاء المصفوفات
A = np.array([10, 20, 30, 40, 50])
B = np.array([5, 4, 3, 2, 1])


# 3. تنفيذ العمليات الحسابية
addition = A + B          # الجمع
subtraction = A - B       # الطرح
multiplication = A * B    # الضرب عنصر بعنصر
division = A / B          # القسمة عنصر بعنصر


# 4. تطبيق دوال numPy
mean_A = np.mean(A)       # المتوسط
max_A = np.max(A)         # أكبر قيمة
min_A = np.min(A)         # أصغر قيمة
dot_product = np.dot(A, B)  # الضرب النقطي
reshaped_A = A.reshape(5, 1)  # إعادة تشكيل المصفوفة


# طباعة النتائج
print("A + B =", addition)
print("A - B =", subtraction)
print("A * B =", multiplication)
print("A / B =", division)

print("Mean of A =", mean_A)
print("Max of A =", max_A)
print("Min of A =", min_A)
print("Dot product of A and B =", dot_product)

print("Reshaped A (5x1):")
print(reshaped_A)


