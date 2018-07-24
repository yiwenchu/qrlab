import json
import types

def encode_complex(obj):
    if type(obj) == types.ComplexType:
        return dict(__complex__=True, re=obj.real, im=obj.imag)
    else:
        raise TypeError

def decode_complex(obj):
    if '__complex__' in obj:
        return obj['re']+1j*obj['im']
    else:
        return obj

def dump(*args, **kwargs):
    '''
    Wrapper for json.dump with additional support for encoding the following:
    - complex values
    And default settings:
    - indent = 4
    - sort_keys = True
    '''
    kwargs['default'] = encode_complex
    kwargs['indent'] = kwargs.get('indent', 4)
    kwargs['sort_keys'] = kwargs.get('sort_keys', True)
    json.dump(*args, **kwargs)

def load(f, **kwargs):
    '''
    Wrapper for json.load with additional support for decoding the following:
    - complex values
    '''
    return json.load(f, object_hook=decode_complex, **kwargs)
