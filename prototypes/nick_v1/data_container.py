
class DataContainer():
    
    def __init__(self):
        self._keystore = {}
    
    def get_item(self, key):
        if key in self.get_keys():
            return self._keystore[key]
        else:
            raise KeyError("provided key '{}' not present in {}".format(key, type(self).__name__))
    
    def set_item(self, key, obj):
        if type(key) != str:
            raise ValueError("provided key must be string type")
        self._keystore[key] = obj
    
    def get_keys(self):
        return self._keystore.keys()
