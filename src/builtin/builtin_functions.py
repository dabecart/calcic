"""
builtin_types.py

Special functions which cannot be declared on code files and are built in the compiler.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from src.global_context import GlobalContext
from src.builtin.builtin_functions_parser import *
from src.builtin.builtin_functions_TAC import *

class BuiltInFunctions:
    FUNCS: dict[str, type]  = {
        # <stdarg.h>
        BuiltIn_va_start.FUNC_NAME  : BuiltIn_va_start,
        BuiltIn_va_arg.FUNC_NAME    : BuiltIn_va_arg,
        BuiltIn_va_end.FUNC_NAME    : BuiltIn_va_end,
        BuiltIn_va_copy.FUNC_NAME   : BuiltIn_va_copy,

        # <stddef.h>
        BuiltIn_offsetof.FUNC_NAME  : BuiltIn_offsetof,

        # Custom macros.
        BuiltIn_asm.FUNC_NAME       : BuiltIn_asm,
    }

    @staticmethod
    def connectHandlersToContext(ctx: GlobalContext):
        ctx.isBuiltInFunctionByIdentifier = BuiltInFunctions.isBuiltInFunctionByIdentifier
        ctx.createBuiltInFunction = BuiltInFunctions.parseBuiltInASTFunctionCall
        ctx.isBuiltInFunctionByClass = BuiltInFunctions.isBuiltInFunctionByClass
        ctx.parseTACBuiltInFunction = BuiltInFunctions.parseBuiltInTACFunctionCall

    @staticmethod
    def isBuiltInFunctionByIdentifier(funcName: str) -> bool:
        return funcName in BuiltInFunctions.FUNCS
    
    @staticmethod
    def isBuiltInFunctionByClass(elem) -> bool:
        return isinstance(elem, BuiltInFunctionCall)
    
    @staticmethod
    def parseBuiltInASTFunctionCall(parentAST: AST, funcName: str) -> BuiltInFunctionCall:
        astClass = BuiltInFunctions.FUNCS.get(funcName)
        if astClass is None:
            raise ValueError(f"Cannot create built-in function call from {funcName}")
        
        return parentAST.createChild(astClass)
    
    @staticmethod
    def parseBuiltInTACFunctionCall(exp: BuiltInFunctionCall, 
                                    insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        # Add TAC to the class name of exp and call the fromAST static function.
        className = exp.__class__.__name__
        tacName = f"TAC{className}"
        tacClass = globals().get(tacName)

        if tacClass is None:
            raise ValueError(f"Cannot create built-in function TAC from {className}")
        
        return tacClass.fromAST(exp, insts, parent)
