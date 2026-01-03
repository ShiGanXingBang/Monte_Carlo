
class Students:
    def __init__(self, name, city, score,):
        self.name = name
        self.city = city
        self.gender = self.city_from()
        self.score = score
        self.grade = self.getgrade()

    def getgrade(self):
        if self.score >= 90:
            return 'A'
        else:
            return 'B'
    def city_from(self):
        if self.city == 'ChengDu':
            return 'male'
        else:
            return 'female'


Sun = Students('Sun', 'ChengDu', 90, )
print(Sun.name , Sun.score , Sun.gender , Sun.city)
