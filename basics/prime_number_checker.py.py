# mohamed ashraf salah
#Prime Number Checker  

#check if a number is prime
def is_prime(n):
    if n < 2:
        return False   
    #بنفحص الرقم من 2 لحد جزر بتاعه
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True