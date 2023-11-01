import json 


class Transaction():
    
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, Class, timestamp):

        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.Class = Class
        self.timestamp = timestamp
    
    def get_transaction(self):
        
        transaction = json.dumps(self.__dict__, sort_keys=True)
        
        return transaction
