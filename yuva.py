students=[]
n=int(input("Enter a number of students: "))
for i in range(n):
   print(f"\nEnter details for student{i+1}: ")
   name=input("Enter name:")
   roll_number=int(input("Enter the roll number: "))
   marks=int(input("Enter marks:"))
   students.append({"name1":name,"roll number":roll_number,"marks":marks})
print("\nStudents with marks less than 60: ")
for student in students:
    if student["marks"]<60:
        print(f"Name:{student['name']},Roll number:{student['roll number']},marks:{student['marks']}")
print(students)