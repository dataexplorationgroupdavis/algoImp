# python3
# an example of object oriented programming in python
class Student:
    '''
    hello world
    '''
    type = 'student' # class variable

    def __init__(self, name, age, gender): # special method
        self.name = name # attributes, instance variable
        self.age = age
        self.gender = gender

    def __str__(self): # making the class a string for printing
        return(('Name: {}, Age: {}, Gender: {}').format(self.name, self.age, self.gender))

    # regular method
    def addGrade(self, course, grade): # method
        print('{}: {}'.format(course, grade))

    @classmethod
    def cfunc(self):
        pass

    @staticmethod
    def sfunc():
        pass

class ClubMember(Student): # subclass
    type = 'club member'
    
    class Club:
        name = 'DEG' # class variable
        def __init__(self): 
            self.name = 'DEG' # instance variable

    def __init__(self, name, age, gender, title):
        super().__init__(name, age, gender) # passing arguments to parent class
        self.title = title

Jack = ClubMember('Jack', 22, 'male', 'member')
Jack.addGrade('Math', 'D')
print(type(Jack))
print(Jack)
print(Jack.age)
print(Jack.type)
print(Jack.title)
print(Jack.Club.name) # class for the class
print(Jack.Club().name) # __init__ for different object
Chris = Student('Chris', '19', 'male')
print(Chris.type)
