"""Colab CLI timeout patch (loaded via .pth at interpreter startup).

google.auth.transport.requests.AuthorizedSession.request defaults
timeout=120s, shorter than TPU v5e-1 provisioning. colab_cli calls
session.request() with no timeout, inheriting 120s, so `colab new` gives
up before the assignment is ready and never saves the token.
"""
import functools

LONG_TIMEOUT = 900

try:
    from google.auth.transport.requests import AuthorizedSession

    _orig = AuthorizedSession.request
    if not getattr(_orig, "_colab_long_timeout", False):

        @functools.wraps(_orig)
        def _request(self, method, url, *args, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = LONG_TIMEOUT
            return _orig(self, method, url, *args, **kwargs)

        _request._colab_long_timeout = True
        AuthorizedSession.request = _request
except Exception:
    pass
