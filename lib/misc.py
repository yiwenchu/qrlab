import os
import shutil

def copy_template(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        print 'Creating %s from %s; you might want to change it' % (dst, src)
        shutil.copy(src, dst)
        return True
    else:
        return False
