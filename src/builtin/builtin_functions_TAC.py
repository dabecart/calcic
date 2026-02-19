"""
builtin_functions_TAC.py

Handles the TAC side of built-in functions.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from src.TAC import *
from src.builtin.builtin_functions_parser import *

class TACBuiltInFunction(TACInstruction):
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
    def __init__(self, param_ap: TACValue, param_parmN: TACValue, 
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.param_ap = param_ap
        self.param_parmN = param_parmN
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> TACValue:
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def printBuiltIn(self) -> str:
        return f"va_start\n"

    @staticmethod
    def fromAST(ast: BuiltIn_va_start, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        param_ap = parent.parseTACExpression(ast.param_ap, insts).convert()
        param_parmN = parent.parseTACExpression(ast.param_parmN, insts).convert()

        func = parent.createChild(TACBuiltIn_va_start, param_ap, param_parmN, insts)

        return TACBaseOperand(func.result, ast.typeId, insts)
    
class TACBuiltIn_va_arg(TACBuiltInFunction):
    def __init__(self, param_ap: TACValue, param_type: DeclaratorType, 
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.param_ap = param_ap
        self.param_type = param_type
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> TACValue:
        # This is where the popped argument from the vargs will be stored. 
        return TACValue(False, self.param_type)

    def printBuiltIn(self) -> str:
        return f"va_arg\n"

    @staticmethod
    def fromAST(ast: BuiltIn_va_arg, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        param_ap = parent.parseTACExpression(ast.param_ap, insts).convert()

        func = parent.createChild(TACBuiltIn_va_arg, param_ap, ast.param_type, insts)

        return TACBaseOperand(func.result, ast.typeId, insts)

class TACBuiltIn_va_end(TACBuiltInFunction):
    def __init__(self, param_ap: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.param_ap = param_ap
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> TACValue:
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def printBuiltIn(self) -> str:
        return f"va_end\n"

    @staticmethod
    def fromAST(ast: BuiltIn_va_end, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        param_ap = parent.parseTACExpression(ast.param_ap, insts).convert()

        func = parent.createChild(TACBuiltIn_va_end, param_ap, insts)

        return TACBaseOperand(func.result, ast.typeId, insts)

