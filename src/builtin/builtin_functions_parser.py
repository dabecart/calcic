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
    
class BuiltIn_offsetof(BuiltInFunctionCall):
    FUNC_NAME: str = "__builtin_offsetof"

    def parseArguments(self, *args) -> DeclaratorType:
        # offsetof(type, member)

        # Parse the "type" argument. This is also the return type.
        _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        if self.peek().id != ",":
            declarator = self.createChild(TopAbstractDeclarator)
            info: DeclaratorInformation = declarator.process(declType)
            self.param_type = info.type
        else:
            self.param_type = declType

        if not isinstance(self.param_type, BaseDeclaratorType) or \
           self.param_type.baseType.name != "STRUCT":
            self.raiseError(f"Invalid type {self.param_type}, expected a struct type")

        self.expect(",")

        # Create a dummy variable of type 'param_type'.
        name = ".dummy."
        mangledName = self.context.mangleIdentifier(name)
        self.context.addVariableIdentifier(name, mangledName, self.param_type)
        var = self.createChild(Variable, name)

        # Now, parse the subsequent terms until the '(' like we were operating on var. Keep adding 
        # the offsets.
        memberElement = self.createChild(Dot, var)
        if memberElement.memberInfo is None:
            raise ValueError()
        self.memberOffset = memberElement.memberInfo.offset
        
        while True:
            postTok = self.peek()
            if postTok.id == "[":
                # This is a subscript.
                memberElement = self.createChild(Subscript, memberElement)
                # Evaluate the array index.
                arrayIndex = memberElement.index.staticEval().getIntegerValue(self)
                # Multiply by the size of the returning object to get the offset.
                self.memberOffset += arrayIndex * memberElement.typeId.getByteSize()

            elif postTok.id == ".":
                # Structure/union dot.
                self.pop()
                memberElement = self.createChild(Dot, memberElement)
                if memberElement.memberInfo is None:
                    raise ValueError()
                self.memberOffset += memberElement.memberInfo.offset

            elif postTok.id == "->":
                # Pointer structure/union arrow.
                self.pop()
                memberElement = self.createChild(Arrow, memberElement)
                if memberElement.memberInfo is None:
                    raise ValueError()
                self.memberOffset += memberElement.memberInfo.offset

            else:
                # No postfix.
                break

        # Remove the dummy variable.
        del self.context.identifierMap[name]
        del self.context.variablesMap[mangledName]

        # Return type is size_t (long).
        # TODO: This is in x64!
        return TypeSpecifier.ULONG.toBaseType()

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}offsetof\n'
    
"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Custom macros.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
class BuiltIn_asmIOType(enum.Enum):
    REGISTER = enum.auto()
    IMMEDIATE = enum.auto()

@dataclass
class BuiltIn_asmIO:
    ioType: BuiltIn_asmIOType
    # Args is used to store the requested register for a REGISTER type.
    args: str
    exp: Exp

    # This needs to be set from the assembler.
    _asmbRepresentation: str = ""

    def setAsmbRepresentation(self, repr: str):
        self._asmbRepresentation = repr.lstrip(" \t\r\n").rstrip(" \t\r\n")

class BuiltIn_asm(BuiltInFunctionCall):
    FUNC_NAME: str = "__asm__"

    def parseIOList(self, validIOTypes: set[BuiltIn_asmIOType]) -> list[BuiltIn_asmIO]:
        # "r:register" (exp)   <- Pass a value to a register before inserting the code.
        # "i" (exp)            <- Replace a value as an immediate in the assembly code.

        ret: list[BuiltIn_asmIO] = []
        while True:
            self.pop() # Either pop the , or the :

            typeStr = self.createChild(String).value

            match typeStr[0].lower():
                case "r":
                    asmIOType = BuiltIn_asmIOType.REGISTER
                    args = typeStr.split(":")
                    if len(args) != 2:
                        self.raiseError(f'Invalid type string {typeStr}. For registers, use "r:<reg>"')
                    arg = args[1]

                case "i":
                    asmIOType = BuiltIn_asmIOType.IMMEDIATE
                    arg = ""

                case _:
                    self.raiseError(f"Invalid type string {typeStr}")

            if asmIOType not in validIOTypes:
                self.raiseError(f"Invalid IO type for this argument")

            self.expect("(")
            exp = self.createChild(Exp)

            if not exp.typeId.isScalar():
                self.raiseError(f"Expected a scalar type")

            self.expect(")")

            asmIO = BuiltIn_asmIO(asmIOType, arg, exp)
            ret.append(asmIO)

            if self.peek().id != ",":
                break

        return ret

    def parseArguments(self, *args) -> DeclaratorType:
        # __asm__ (asmbCode : outputs : inputs);
        self.asmbCode = self.createChild(String).value

        self.outputs: list[BuiltIn_asmIO] = []
        if self.peek().id == ":":
            self.outputs = self.parseIOList(set([BuiltIn_asmIOType.REGISTER]))
            for output in self.outputs:
                if not output.exp.isLvalueAssignable():
                    self.raiseError(f"Expected an lvalue as argument for {output.args}")

        self.inputs: list[BuiltIn_asmIO]  = []
        if self.peek().id == ":":
            self.inputs = self.parseIOList(set([BuiltIn_asmIOType.IMMEDIATE, BuiltIn_asmIOType.REGISTER]))

        return TypeSpecifier.VOID.toBaseType()
    
    # Before calling this, remember to set all _asmbRepresentation of the inputs.
    def generateAssemblyCode(self) -> str:
        # Replace the %\d with the immediate representation.
        for index, input in enumerate(self.inputs):
            if input.ioType == BuiltIn_asmIOType.IMMEDIATE:
                self.asmbCode = self.asmbCode.replace(f"%{index}", input._asmbRepresentation)
        
        # Replace all %% with %.
        self.asmbCode = self.asmbCode.replace("%%", "%")
        return self.asmbCode

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}__asm__\n'