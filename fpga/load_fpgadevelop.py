import os
import sys
import mclient

try:
    import YngwieEncoding
except:
    yp = os.path.join(mclient.get_source_dir(), 'Yngwie/Python/Core')
    sys.path.append(yp)
    import YngwieEncoding

import YngwieInterface
import YngwieDecoding
import BitArray
import BooleanExpression
