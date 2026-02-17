"""
builtin_functions_TAC.py

Handles the TAC side of built-in functions.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from ..TAC import *

class TACBuiltInFunction(TACInstruction):
    def __init__(self, arguments: list[TACValue], instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.arguments = arguments
        super().__init__(instructionsList, parentTAC)
    
    @abstractmethod
    def parse(self) -> TACValue:
        pass

    @staticmethod
    @abstractmethod
    def fromAST(ast: AST, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        pass

    @abstractmethod
    def printBuiltIn(self) -> str:
        pass

    def print(self) -> str:
        return f"Built-in FunctionCall: {self.printBuiltIn()}"

class TACBuiltIn_va_start(TACBuiltInFunction):
    def parse(self) -> TACValue:
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def printBuiltIn(self) -> str:
        return f"va_start\n"

    @staticmethod
    def fromAST(ast: AST, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        parent.createChild(TACBuiltIn_va_start, None, insts)

        val = TACValue(False, TypeSpecifier.VOID.toBaseType())
        return TACBaseOperand(val, TypeSpecifier.VOID.toBaseType(), insts)
    
class TACBuiltIn_va_arg(TACBuiltInFunction):
    def parse(self) -> TACValue:
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def printBuiltIn(self) -> str:
        return f"va_arg\n"

    @staticmethod
    def fromAST(ast: AST, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        parent.createChild(TACBuiltIn_va_arg, None, insts)

        val = TACValue(False, TypeSpecifier.VOID.toBaseType())
        return TACBaseOperand(val, TypeSpecifier.VOID.toBaseType(), insts)

class TACBuiltIn_va_end(TACBuiltInFunction):
    def parse(self) -> TACValue:
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def printBuiltIn(self) -> str:
        return f"va_end\n"

    @staticmethod
    def fromAST(ast: AST, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        parent.createChild(TACBuiltIn_va_end, None, insts)

        val = TACValue(False, TypeSpecifier.VOID.toBaseType())
        return TACBaseOperand(val, TypeSpecifier.VOID.toBaseType(), insts)

