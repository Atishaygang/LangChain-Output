from pydantic import BaseModel , EmailStr , Field
from typing import Optional 

class Student(BaseModel):
    name:str
    age:Optional[int] = None
    email:EmailStr = 'abc@gmail.com'
    cgpa: float = Field(gt=0 , lt= 10 , description='A number that signify marks of a student',default=5)

new_student = { 'name':'Atishay',}
student = Student(**new_student)

student_dict = dict(student)
student_json = student.model_dump_json()

print(student_dict)

