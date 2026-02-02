#mohamed ashraf salah
#1-class and object
class book:
    def __init__(self, title, author, isbn, available=True):    
        self.title = title
        self.author = author
        self.isbn = isbn
        self.available = available
#3-Encapsulation
    def get_isbn(self):
        return self.__isbn
#3-Encapsulation
    def set_isbn(self, new_isbn):
        self.__isbn = new_isbn 
#1-class and object
    def display_info(self):
        status = "Available" if self.available else "Not Available"
        print(f"Title: {self.title}, Author: {self.author}, ISBN: {self.__isbn}, Status: {status}")

class Member:
    def __init__(self, name, membership_id):
        self.name = name
        self.member_id = membership_id
        self.borrowed_books = []

    def get_membership_id(self):
        return self.__membership_id
    
    def set_membership_id(self, new_id):
        self.__membership_id = new_id


    def borrow_book(self, book):
        if book.available:
            self.borrowed_books.append(book)
            book.available = False
            print(f"{self.name} borrowed '{book.title}'.")
        else:
            print(f"Sorry, '{book.title}' is not available.")

    def return_book(self, book):
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            book.available = True
            print(f"{self.name} returned '{book.title}'.")
        else:
            print(f"{self.name} doesn't have this book.")

class StaffMember(Member):
    def __init__(self, name, membership_id, staff_id):
        super().__init__(name, membership_id)
        self.staff_id = staff_id

    def add_book(self, library, title, author, isbn):
        new_book = book(title, author, isbn)
        library.books.append(new_book)
        print(f"Staff {self.name} added book '{title}' to the library.")

class Library:
    def __init__(self):
        self.books = []

  
    def show_books(self):
        print("\nLibrary Books:")
        for book in self.books:
            book.display_info()

library = Library()


staff = StaffMember("Ahmed", 101, "S-01")
staff.add_book(library, "Python Basics", "John Doe", "12345")
staff.add_book(library, "Data Science", "Jane Smith", "98765")


library.show_books()


member = Member("Mohamed", 202)
member.borrow_book(library.books[0])


library.show_books()

member.return_book(library.books[0])
library.show_books()
    

    