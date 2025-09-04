from torchapp.testing import TorchAppTestCase
from psyop.apps import Psyop


class TestPsyop(TorchAppTestCase):
    app_class = Psyop
