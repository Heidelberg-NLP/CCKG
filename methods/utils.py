import json

class MyJsonSerializer(json.JSONEncoder):
    """
    Custom encoder for json serialization
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

        try:
            return super().default(obj)
        except TypeError:  # needed for datahandlers
            return obj._to_json()
