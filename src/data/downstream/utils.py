    
    
    
    
    
def keep_number(text:str) -> int:
    tokens = text.split('\\')
    number = tokens[-1]
    number = number[0:-4]
    return number