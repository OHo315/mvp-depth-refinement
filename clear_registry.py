# Small script to delete the registered keys in mmengine in an attempt to start with a fresh dict containing model and dataset classes.
# However, after further debugging, this was not the issue.

import sys

for k in list(sys.modules.keys()):
    if k.startswith("estimator."):
        del sys.modules[k]