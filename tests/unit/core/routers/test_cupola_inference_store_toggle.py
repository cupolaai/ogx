# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: CUPOLA_INFERENCE_STORE=off gives the inference router no store."""

import pytest

from ogx.core.routers.inference import InferenceRouter

STORE = object()


@pytest.mark.parametrize("value", ["off", "OFF", " off "])
def test_off_disables_the_store(monkeypatch, value):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", value)
    assert InferenceRouter(None, store=STORE).store is None


@pytest.mark.parametrize("value", [None, "", "on", "false", "0", "offline"])
def test_anything_else_keeps_the_store(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUPOLA_INFERENCE_STORE", raising=False)
    else:
        monkeypatch.setenv("CUPOLA_INFERENCE_STORE", value)
    assert InferenceRouter(None, store=STORE).store is STORE
