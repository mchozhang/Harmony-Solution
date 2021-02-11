# flask wsgi
import sys

sys.path.insert(0, '/var/www/harmony-solution')
from game.app import app as application