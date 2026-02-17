"""
global_context.py

Values used in all the stages of the compiler.

calcic. Written by @dabecart, 2026.
"""

from dataclasses import dataclass, field
from typing import Callable
from .types import *
from .builtin.builtin_types import BuiltInTypes

@dataclass
class GlobalContext:
    builtInTypes: BuiltInTypes                               = field(default_factory=lambda: BuiltInTypes())
    isBuiltInFunctionByIdentifier: Callable                  = lambda *args, **kwargs: None
    createBuiltInFunction: Callable                          = lambda *args, **kwargs: None
    isBuiltInFunctionByClass: Callable                       = lambda *args, **kwargs: None
    parseTACBuiltInFunction: Callable                        = lambda *args, **kwargs: None
    isBuiltInTACFunction: Callable                           = lambda *args, **kwargs: None

globalContext = GlobalContext()