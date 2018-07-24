import sys
import cPickle as pickle
import base64
import subprocess

def start_python_process(pyfile, **kwargs):
    '''
    Start an independent python process executing pyfile.
    Keyword arguments are passed to the process directly through the Popen
    call and can be retrieved by the subprocess using the 'parse_args'
    function.
    '''
    s = base64.b64encode(pickle.dumps(kwargs))
    cmd = [sys.executable, pyfile, '--kwargs', s]
    pid = subprocess.Popen(cmd).pid
    return pid

def decode_kwargs(s):
    kwargs = pickle.loads(base64.b64decode(s))
    return kwargs

class ArgParser(object):

    def __init__(self, description=None):
        self._argtypes = {}
        self._description = description
        self._args = list()
        self._kwargs = dict()

    def __getattr__(self, name):
        if name in self._kwargs:
            return self._kwargs[name]
        raise ValueError('Option %s not set' % (name,))

    def add_option(self, name, type=str, default=None, help=None):
        if name.startswith('--'):
            name = name[2:]
        self._argtypes[name] = dict(type=type, default=default, help=help)
        self._kwargs[name] = default

    def parse_args(self, s=None):
        '''
        Parse command line arguments passed by 'start_python_process' and return
        a dictionary.
        '''
        if s is not None:
            return decode_kwargs(s)

        i = 1
        while i < len(sys.argv):

            if sys.argv[i] == '--kwargs':
                if not i < len(sys.argv):
                    raise ValueError('Expecting argument')
                self._kwargs.update(decode_kwargs(sys.argv[i+1]))
                i += 2

            elif sys.argv[i].startswith('--'):
                if not i < len(sys.argv):
                    raise ValueError('Expecting argument')
                name = sys.argv[i][2:]
                val = sys.argv[i+1]
                if name in self._argtypes:
                    val = self._argtypes[name]['type'](val)
                if name != '':
                    self._kwargs[name] = val
                i += 2

            else:
                self._args.append(sys.argv[i])
                i += 1

        return self._args, self._kwargs

def parse_args():
    ap = ArgParser()
    return ap.parse_args()

# Test by starting ourselves in a new process
if __name__ == '__main__':
    args, kwargs = parse_args()
    if len(kwargs) == 0:
        start_python_process(__file__,
            test=['an array', 'with', 'some', 'elements'],
            test2={'a': 'dictionary'})
    else:
        print 'Arguments: %s, Keyword arguments: %s' % (args, kwargs, )


