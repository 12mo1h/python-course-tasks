# mohamed ashraf salah
#Student Grades-list of dicts

#انشئ list of dicts
empty_dict = {}
students= [{"name":"Ahmed","grade":[100,90,55],},
          {"name":"Sara","grade":[85,95,80],},
          {"name":"Omar","grade":[70,75,90],}
          ]

# اطبع أسماء الطلاب مع متوسط درجاتهم
for student in students:
    average_grade = sum(student["grade"]) / len(student["grade"])
    print(f"{student['name']}'s average grade is: {average_grade}")
              