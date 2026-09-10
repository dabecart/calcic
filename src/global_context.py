"""
global_context.py

Values used in all the stages of the compiler.

calcic. Written by @dabecart, 2026.
"""

from dataclasses import dataclass, field
from typing import Callable
import enum

from src.calcic_types import *
from src.builtin.builtin_types import BuiltInTypes

class TargetArchitectures(enum.Enum):
    x64     = "x64"
    calci32 = "calci32"

@dataclass
class GlobalContext:
    targetArchitecture: TargetArchitectures                 = TargetArchitectures.x64
    addressByteLen: int                                     = 8
    linkerRoute: str                                        = "gcc"
    
    useGCCLibraries: bool                                   = False
    useCalcicSTDLibraries: bool                             = True
    # To know when to add the entry point to the generated assembly.
    generateExecutable: bool                                = True

    builtInTypes: BuiltInTypes                              = field(default_factory=lambda: BuiltInTypes())
    isBuiltInFunctionByIdentifier: Callable                 = lambda *args, **kwargs: None
    createBuiltInFunction: Callable                         = lambda *args, **kwargs: None
    isBuiltInFunctionByClass: Callable                      = lambda *args, **kwargs: None
    parseTACBuiltInFunction: Callable                       = lambda *args, **kwargs: None

    # Debug information.
    addDebugInfo: bool                                      = False

    def setArchitecture(self, arch: TargetArchitectures):
        self.targetArchitecture = arch

        match arch:
            case TargetArchitectures.x64:
                self.addressByteLen = 8
                self.linkerRoute = "gcc"
            case TargetArchitectures.calci32:
                self.addressByteLen = 4
                self.linkerRoute = "./linker/calcil/calcil"
            case _:
                raise ValueError(f"Invalid architecture: {arch}")

globalContext = GlobalContext()

# If running the 'Writing a C compiler' testsuite...
globalContext.useGCCLibraries = True
globalContext.useCalcicSTDLibraries = False
globalContext.generateExecutable = False