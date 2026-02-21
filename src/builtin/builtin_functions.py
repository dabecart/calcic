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
    FUNCS: list[str] = [
        BuiltIn_va_start.FUNC_NAME,
        BuiltIn_va_arg.FUNC_NAME,
        BuiltIn_va_end.FUNC_NAME,
        BuiltIn_va_copy.FUNC_NAME,
    ]

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
        match funcName:
            # <stdarg.h>
            case BuiltIn_va_start.FUNC_NAME:  return parentAST.createChild(BuiltIn_va_start)
            case BuiltIn_va_arg.FUNC_NAME:    return parentAST.createChild(BuiltIn_va_arg)
            case BuiltIn_va_end.FUNC_NAME:    return parentAST.createChild(BuiltIn_va_end)
            case BuiltIn_va_copy.FUNC_NAME:    return parentAST.createChild(BuiltIn_va_copy)
            case _:
                raise ValueError(f"Cannot create builtin function {funcName}")
    
    @staticmethod
    def parseBuiltInTACFunctionCall(exp: BuiltInFunctionCall, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        match exp:
            # <stdarg.h>
            case BuiltIn_va_start():    return TACBuiltIn_va_start.fromAST(exp, insts, parent)
            case BuiltIn_va_arg():      return TACBuiltIn_va_arg.fromAST(exp, insts, parent)
            case BuiltIn_va_end():      return TACBuiltIn_va_end.fromAST(exp, insts, parent)
            case BuiltIn_va_copy():     return TACBuiltIn_va_copy.fromAST(exp, insts, parent)
            case _:
                raise ValueError(f"Cannot create built-in function TAC from {exp}")

