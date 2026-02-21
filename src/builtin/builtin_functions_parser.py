"""
builtin_functions_parser.py

Handles the parsing side of built-in functions.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from src.parser import *

class BuiltInFunctionCall(Exp):
    # To override.
    FUNC_NAME: str = ""

    def parse(self, *args):
        self.functionName = self.expect("identifier").value
        if self.functionName != self.FUNC_NAME:
            self.raiseError(f"Function name {self.functionName} doesn't match {self.FUNC_NAME}")

        self.expect("(")

        self.typeId = self.parseArguments(args)

        self.expect(")")

    @abstractmethod
    def parseArguments(self, *args) -> DeclaratorType:
        pass

    @abstractmethod
    def print(self, padding: int) -> str:
        return super().print(padding)

    def staticEval(self) -> StaticEvalValue:
        self.raiseError("Cannot evaluate a function call during compilation")        
    
"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
<stdargs.h>
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
class BuiltIn_va_start(BuiltInFunctionCall):
    FUNC_NAME: str = "__builtin_va_start"

    def parseArguments(self, *args) -> DeclaratorType:
        # void va_start(va_list ap, parmN);
        if not self.context.functionMap[self.context.insideFunctionName].variadic:
            self.raiseError("Cannot use va_start in non-variadic function")

        self.param_ap: Exp = self.createChild(Exp).preconvertExpression()

        # See if this argument needs a cast before being passed to the function.
        vaListTypeDecayed = globalContext.builtInTypes.va_list.decay()
        if self.param_ap.typeId != vaListTypeDecayed:
            self.param_ap = self.createChild(
                Cast, vaListTypeDecayed, self.param_ap, True).preconvertExpression()

        self.expect(",")

        self.param_parmN: Exp = self.createChild(Exp).preconvertExpression()

        return TypeSpecifier.VOID.toBaseType()

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}va_start\n'
    
class BuiltIn_va_arg(BuiltInFunctionCall):
    FUNC_NAME: str = "__builtin_va_arg"

    def parseArguments(self, *args) -> DeclaratorType:
        # type va_arg(va_list ap, type);

        self.param_ap: Exp = self.createChild(Exp).preconvertExpression()

        # See if this argument needs a cast before being passed to the function.
        vaListTypeDecayed = globalContext.builtInTypes.va_list.decay()
        if self.param_ap.typeId != vaListTypeDecayed:
            self.param_ap = self.createChild(
                Cast, vaListTypeDecayed, self.param_ap, True).preconvertExpression()

        self.expect(",")
            
        # Parse the "type" argument. This is also the return type.
        _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        if self.peek().id != ")":
            declarator = self.createChild(TopAbstractDeclarator)
            info: DeclaratorInformation = declarator.process(declType)
            self.param_type = info.type
        else:
            self.param_type = declType

        if not self.param_type.isComplete():
            self.raiseError(f"{self.param_type} is not a complete type")

        if isinstance(self.param_type, (ArrayDeclaratorType, FunctionDeclaratorType)):
            self.raiseError(f"The second argument to va_arg cannot be a {self.param_type}")

        return self.param_type

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}va_arg\n'
    
class BuiltIn_va_end(BuiltInFunctionCall):
    FUNC_NAME: str = "__builtin_va_end"

    def parseArguments(self, *args) -> DeclaratorType:
        # void va_end(va_list ap);

        self.param_ap: Exp = self.createChild(Exp).preconvertExpression()

        # See if this argument needs a cast before being passed to the function.
        vaListTypeDecayed = globalContext.builtInTypes.va_list.decay()
        if self.param_ap.typeId != vaListTypeDecayed:
            self.param_ap = self.createChild(
                Cast, vaListTypeDecayed, self.param_ap, True).preconvertExpression()

        return TypeSpecifier.VOID.toBaseType()

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}va_end\n'
    
class BuiltIn_va_copy(BuiltInFunctionCall):
    FUNC_NAME: str = "__builtin_va_copy"

    def parseArguments(self, *args) -> DeclaratorType:
        # void va_end(va_list ap);

        self.param_dest: Exp = self.createChild(Exp).preconvertExpression()

        # See if this argument needs a cast before being passed to the function.
        vaListTypeDecayed = globalContext.builtInTypes.va_list.decay()
        if self.param_dest.typeId != vaListTypeDecayed:
            self.param_dest = self.createChild(
                Cast, vaListTypeDecayed, self.param_dest, True).preconvertExpression()

        self.expect(",")

        self.param_src: Exp = self.createChild(Exp).preconvertExpression()

        if self.param_src.typeId != vaListTypeDecayed:
            self.param_src = self.createChild(
                Cast, vaListTypeDecayed, self.param_src, True).preconvertExpression()

        return TypeSpecifier.VOID.toBaseType()

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}va_copy\n'