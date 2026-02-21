"""
builtin_functions_TAC.py

Handles the TAC side of built-in functions.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from src.TAC import *
from src.builtin.builtin_functions_parser import *

class TACBuiltInFunction(TACInstruction):
    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TAC
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
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

    """
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TAC OPTIMIZER
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    @abstractmethod
    def anotateReachingCopies(self, copies: set[TACCopy], aliased: set[TACValue]):
        pass

    @abstractmethod
    def rewriteWithReachingCopies(self) -> TACInstruction|None:
        pass

    @abstractmethod
    def anotateLiveVariables(self, liveVariables: set[TACValue], aliased: set[TACValue]):
        pass

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
<stdargs.h>
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
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

    def anotateReachingCopies(self, copies: set[TACCopy], aliased: set[TACValue]):
        # This function only affects ap and generates no result. It shouldn't affect aliased 
        # variables.
        copiesToRemove: set[TACCopy] = set(
            c
            for c in copies
            if self.param_ap in (c.src, c.dst)
        )
        copies -= copiesToRemove

    def rewriteWithReachingCopies(self) -> TACInstruction|None:
        isReplaceable, newParam_ap = self.replaceOperand(self.param_ap)
        if isReplaceable:
            return TACBuiltIn_va_start(newParam_ap, self.param_parmN, self.insts, self.parent)
        return self

    def anotateLiveVariables(self, liveVariables: set[TACValue], aliased: set[TACValue]):
        # Both ap and parmN are alive before the call to va_start.
        liveVariables |= {self.param_ap, self.param_parmN}

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

    def anotateReachingCopies(self, copies: set[TACCopy], aliased: set[TACValue]):
        # This function affects ap and generates a result. It shouldn't affect aliased 
        # variables.
        copiesToRemove: set[TACCopy] = set(
            c
            for c in copies
            if self.param_ap in (c.src, c.dst) or self.result in (c.src, c.dst)
        )
        copies -= copiesToRemove

    def rewriteWithReachingCopies(self) -> TACInstruction|None:
        isReplaceable, newParam_ap = self.replaceOperand(self.param_ap)
        if isReplaceable:
            ret = TACBuiltIn_va_arg(newParam_ap, self.param_type, self.insts, self.parent)
            ret.result = self.result
            return ret
        return self

    def anotateLiveVariables(self, liveVariables: set[TACValue], aliased: set[TACValue]):
        # Kill the result.
        if self.result in liveVariables:
            liveVariables.remove(self.result)
        
        # ap is alive before the call to va_start.
        liveVariables |= {self.param_ap}

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

    def anotateReachingCopies(self, copies: set[TACCopy], aliased: set[TACValue]):
        # This function only affects ap and generates no result. It shouldn't affect aliased 
        # variables.
        copiesToRemove: set[TACCopy] = set(
            c
            for c in copies
            if self.param_ap in (c.src, c.dst)
        )
        copies -= copiesToRemove

    def rewriteWithReachingCopies(self) -> TACInstruction|None:
        isReplaceable, newParam_ap = self.replaceOperand(self.param_ap)
        if isReplaceable:
            return TACBuiltIn_va_end(newParam_ap, self.insts, self.parent)
        return self

    def anotateLiveVariables(self, liveVariables: set[TACValue], aliased: set[TACValue]):
        # ap is alive before the call to va_start.
        liveVariables |= {self.param_ap}

class TACBuiltIn_va_copy(TACBuiltInFunction):
    def __init__(self, param_dest: TACValue, param_src: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.param_dest = param_dest
        self.param_src = param_src
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> TACValue:
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def printBuiltIn(self) -> str:
        return f"va_copy\n"

    @staticmethod
    def fromAST(ast: BuiltIn_va_copy, insts: list[TACInstruction], parent: TAC) -> TACExpressionResult:
        param_dest = parent.parseTACExpression(ast.param_dest, insts).convert()
        param_src = parent.parseTACExpression(ast.param_src, insts).convert()

        func = parent.createChild(TACBuiltIn_va_copy, param_dest, param_src, insts)

        return TACBaseOperand(func.result, ast.typeId, insts)

    def anotateReachingCopies(self, copies: set[TACCopy], aliased: set[TACValue]):
        # This function should only affect param_dest and generates no result. It shouldn't affect 
        # other aliased variables.
        copiesToRemove: set[TACCopy] = set(
            c
            for c in copies
            if self.param_dest in (c.src, c.dst)
        )
        copies -= copiesToRemove

    def rewriteWithReachingCopies(self) -> TACInstruction|None:
        isReplaceableDest, newParam_dest = self.replaceOperand(self.param_dest)
        isReplaceableSrc, newParam_src = self.replaceOperand(self.param_src)
        if isReplaceableDest or isReplaceableSrc:
            return TACBuiltIn_va_copy(newParam_dest, newParam_src, self.insts, self.parent)
        return self

    def anotateLiveVariables(self, liveVariables: set[TACValue], aliased: set[TACValue]):
        # Both dest and src are alive before the call to va_start.
        liveVariables |= {self.param_dest, self.param_src}
