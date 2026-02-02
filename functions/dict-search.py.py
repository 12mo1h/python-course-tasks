# mohamed ashraf salah
#dict search


#انشئ قاموسًا dict
empty_dict = {}
contacts = {"Mohamed":"012344445",
            "Ali":"012345678",
            "Omar":"012346789"}


# اطبع القاموس loop
for name in contacts:
    print(name)
#طلب من المستخدم إدخال اسم للبحث عنه
search_name = input("\n Enter a name to search for: ")


# تحقق مما إذا كان الاسم موجودًا في القاموس
if search_name in contacts:
    print(f"The contact number for {search_name} is {contacts[search_name]}")
else:
    print(f"{search_name} is not in the contacts.")






