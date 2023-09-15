
#Import from parent directory so that modules such as "Core" can be located. Just for testing.
import path
import sys
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)