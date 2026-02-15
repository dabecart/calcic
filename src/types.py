"""
types.py

Different C types and declarators used widely in the compiler. 

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
import struct
import math
from typing import TypeVar

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TYPES
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

# Util to store parameter info, such as those of functions, structs or unions.
@dataclass
class ParameterInformation:
    type: DeclaratorType
    name: str

    # Used for the parameters of structs.
    offset: int = 0

    def __str__(self) -> str:
        return f"{self.type} {self.name}"
    
    def __eq__(self, other):
        if not isinstance(other, ParameterInformation):
            return False
        return self.type == other.type

# The storage class defines the lifetime, scope, and visibility of variables.
class StorageClass(enum.Enum):
    STATIC = "static"
    EXTERN = "extern"
    # TODO: auto, register

# Type qualifiers advise the compiler about how the variable will be used.
@dataclass
class TypeQualifier:
    const: bool = False
    # TODO: volatile, restrict

    def toSet(self) -> set[str]:
        ret = set()
        if self.const: ret.add("const")
        return ret

    @staticmethod
    def fromSet(inputSet: set[str]) -> TypeQualifier:
        return TypeQualifier("const" in inputSet)

    def contains(self, other: TypeQualifier) -> bool:
        return other.toSet() <= self.toSet()
    
    def union(self, other: TypeQualifier) -> TypeQualifier:
        un = self.toSet() | other.toSet()
        return TypeQualifier.fromSet(un)

    def __hash__(self) -> int:
        return hash(self.const)

    def __str__(self) -> str:
        ret = ""
        if self.const: ret += "const "
        return ret

class TypeSpecifier:
    VOID: TypeSpecifier
    CHAR: TypeSpecifier
    SIGNED_CHAR: TypeSpecifier
    UCHAR: TypeSpecifier
    SHORT: TypeSpecifier
    INT: TypeSpecifier
    LONG: TypeSpecifier
    USHORT: TypeSpecifier
    UINT: TypeSpecifier
    ULONG: TypeSpecifier
    DOUBLE: TypeSpecifier
    FLOAT: TypeSpecifier

    @staticmethod
    def STRUCT(originalIdentifier: str, mangledIdentifier: str, byteSize: int, alignment: int, 
               members: list[ParameterInformation]) -> TypeSpecifier:
        kwargs = dict(identifier = mangledIdentifier, members = members)
        return TypeSpecifier("STRUCT", f"struct {originalIdentifier}", 
                             byteSize, alignment, **kwargs)
    
    @staticmethod
    def UNION(originalIdentifier: str, mangledIdentifier: str, byteSize: int, alignment: int, 
               members: list[ParameterInformation]) -> TypeSpecifier:
        kwargs = dict(identifier = mangledIdentifier, members = members)
        return TypeSpecifier("UNION", f"union {originalIdentifier}", 
                             byteSize, alignment, **kwargs)

    @staticmethod
    def ENUM(originalIdentifier: str) -> TypeSpecifier:
        # Enums are represented by an int (its size and alignment are 4).
        return TypeSpecifier("ENUM", f"enum {originalIdentifier}", 4, 4)
    
    def __init__(self, name: str, value: str, byteSize: int, alignment: int, **kwargs) -> None:
        self.name: str = name
        self.value: str = value
        self.byteSize: int = byteSize
        self.alignment: int = alignment
        
        # Struct/union stuff.
        self.identifier: str = kwargs.get("identifier", "") # Mangled identifier.
        self.members: list[ParameterInformation] = kwargs.get("members", [])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeSpecifier):
            return False
        # Use the names only. The value field will conflict with typedefs.
        ret = self.name == other.name
        if ret and self.name in ("STRUCT", "UNION"):
            # Check the struct and union's declaration, in particular, the mangled identifier.
            ret = self.identifier == other.identifier
        return ret
    
    def __str__(self) -> str:
        return self.value
    
    def toBaseType(self, qualifiers = TypeQualifier()) -> BaseDeclaratorType:
        return BaseDeclaratorType(self, qualifiers)

    def getByteSize(self) -> int:
        return self.byteSize
    
    def getAlignment(self) -> int:
        return self.alignment
            
    def getMembers(self) -> list[ParameterInformation]:
        if self.name not in ("STRUCT", "UNION"):
            raise ValueError("Cannot get member of a non struct/union type")

        return self.members
    
    def getMember(self, memberName: str) -> ParameterInformation|None:
        for member in self.getMembers():
            if member.name == memberName:
                return member
        return None

    def isSignedInt(self) -> bool:
        match self:
            case TypeSpecifier.CHAR | TypeSpecifier.SIGNED_CHAR | \
                 TypeSpecifier.SHORT | TypeSpecifier.INT | TypeSpecifier.LONG:      
                return True
            case _:
                return False
            
    def isInteger(self) -> bool:
        return self in (TypeSpecifier.CHAR, TypeSpecifier.UCHAR, TypeSpecifier.SIGNED_CHAR,
                        TypeSpecifier.SHORT, TypeSpecifier.USHORT, 
                        TypeSpecifier.INT, TypeSpecifier.UINT, 
                        TypeSpecifier.LONG, TypeSpecifier.ULONG) or \
               self.name == "ENUM"

    def isDecimal(self) -> bool:
        return self in (TypeSpecifier.DOUBLE, TypeSpecifier.FLOAT)
    
    def isCharacter(self) -> bool:
        return self in (TypeSpecifier.CHAR, TypeSpecifier.UCHAR, TypeSpecifier.SIGNED_CHAR) 
    
    def needsIntegerPromotion(self) -> bool:
        return self in (TypeSpecifier.CHAR, TypeSpecifier.UCHAR, TypeSpecifier.SIGNED_CHAR, 
                        TypeSpecifier.SHORT, TypeSpecifier.USHORT) or \
               self.name == "ENUM"

    def flattenStruct(self) -> list[DeclaratorType]:
        if self.name != "STRUCT":
            raise ValueError(f"Cannot flatten {self.name}")
        
        ret: list[DeclaratorType] = []
        
        # Padding will be defined using void.
        def addPadding(byteCount: int):
            ret.extend([TypeSpecifier.VOID.toBaseType()] * byteCount)
        
        lastOffset: int = 0
        member: ParameterInformation
        for member in self.members:
            addPadding(member.offset - lastOffset)
            lastOffset = member.offset + member.type.getByteSize()

            if isinstance(member.type, BaseDeclaratorType) and member.type.baseType.name == "STRUCT":
                ret.extend(member.type.baseType.flattenStruct())
            elif isinstance(member.type, ArrayDeclaratorType):
                arrayBaseType = member.type.getArrayBaseType()
                totalArrayLength = member.type.getByteSize() // arrayBaseType.getByteSize()
                ret.extend([arrayBaseType] * totalArrayLength)
            else:
                ret.append(member.type)

        # Add the final padding.
        addPadding(self.getByteSize() - lastOffset)

        return ret

TypeSpecifier.VOID          = TypeSpecifier("VOID",         "void",         -1, -1)
TypeSpecifier.CHAR          = TypeSpecifier("CHAR",         "char",          1,  1)
TypeSpecifier.SIGNED_CHAR   = TypeSpecifier("SIGNED_CHAR",  "signed char",   1,  1)
TypeSpecifier.UCHAR         = TypeSpecifier("UCHAR",        "unsigned char", 1,  1)
TypeSpecifier.SHORT         = TypeSpecifier("SHORT",        "short",         2,  2)
TypeSpecifier.USHORT        = TypeSpecifier("USHORT",       "unsigned short",2,  2)
TypeSpecifier.INT           = TypeSpecifier("INT",          "int",           4,  4)
TypeSpecifier.UINT          = TypeSpecifier("UINT",         "unsigned int",  4,  4)
TypeSpecifier.LONG          = TypeSpecifier("LONG",         "long",          8,  8)
TypeSpecifier.ULONG         = TypeSpecifier("ULONG",        "unsigned long", 8,  8)
TypeSpecifier.DOUBLE        = TypeSpecifier("DOUBLE",       "double",        8,  8)
TypeSpecifier.FLOAT         = TypeSpecifier("FLOAT",        "float",         4,  4)


"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DECLARATORS
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

DT = TypeVar("DT", bound="DeclaratorType")

# When parsing the "type" of variable being declared, a DeclaratorType will be used. 
# There are three types of declarators:
# - Base types, which are normal types (int, long, float...).
# - Pointer.
# - Function.
# DeclaratorType stores types in a reverse order to normal C.
# An example, int (*param)[3][4] is read as "param is a POINTER to an ARRAY of 3 of an ARRAY of 4 int"
# DeclaratorType is Array(Array(Pointer(Base(int)), 3), 4)
class DeclaratorType(ABC):
    def __init__(self) -> None:
        # Alias set with typedef.
        self.alias = ""
        super().__init__()

    def setAlias(self, alias: str):
        self.alias = alias

    # Returns a new decayed declarator type (if the innermost type is an array, it is converted to
    # a pointer).
    @abstractmethod
    def decay(self) -> DeclaratorType:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def _internal_str(self) -> str:
        return super().__str__()

    def __str__(self) -> str:
        ret = self._internal_str()
        if self.alias == "": 
            return ret
        return f"{self.alias} ({ret})"

    @abstractmethod
    def unqualify(self) -> BaseDeclaratorType:
        pass

    @abstractmethod
    def getTypeQualifiers(self) -> TypeQualifier:
        pass

    @abstractmethod
    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        pass

    @abstractmethod
    def _internal_copy(self: DT) -> DT:
        pass

    def copy(self: DT) -> DT:
        ret = self._internal_copy()
        ret.alias = self.alias
        return ret

    def __repr__(self) -> str:
        return self.__str__()

    def isInteger(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.isInteger()
        return False

    def isDecimal(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.isDecimal()
        return False

    def getByteSize(self) -> int:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.getByteSize()
        elif isinstance(self, PointerDeclaratorType):
            # TODO: For 64 bit systems...
            return 8
        elif isinstance(self, ArrayDeclaratorType):
            return self.declarator.getByteSize() * self.size

        raise ValueError("Not implemented")
    
    def getAlignment(self) -> int:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.getAlignment()
        elif isinstance(self, PointerDeclaratorType):
            # TODO: For 64 bit systems...
            return 8
        elif isinstance(self, ArrayDeclaratorType):
            # ABI says that for arrays bigger or equal to 16 bytes, the alignment is 16. 
            # return 16 if self.getByteSize() >= 16 else self.declarator.getAlignment()
            # TODO: is the rule above necessary for structs?
            return self.declarator.getAlignment()

        raise ValueError("Not implemented")

    def isSigned(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.isSignedInt()
        return False
    
    # Traverses an array type 
    def getArrayBaseType(self) -> DeclaratorType:
        if isinstance(self, ArrayDeclaratorType):
            return self.declarator.getArrayBaseType()
        else:
            return self
    
    def isComplete(self) -> bool:
        # Void is always incomplete.
        if isinstance(self, BaseDeclaratorType) and self.baseType == TypeSpecifier.VOID:
            return False
        # If the byte size cannot be calculated or is negative, then it's an incomplete type.
        try:
            byteSize = self.getByteSize()
        except:
            return False
        return byteSize >= 0
    
    def isScalar(self) -> bool:
        if isinstance(self, (ArrayDeclaratorType, FunctionDeclaratorType)) or \
           self == TypeSpecifier.VOID.toBaseType() or \
           (isinstance(self, BaseDeclaratorType) and self.baseType.name in ("STRUCT", "UNION")):
            return False
        return True
    
    def isArithmetic(self) -> bool:
        if isinstance(self, BaseDeclaratorType):
            return self.baseType.name not in ("STRUCT", "UNION", "VOID")
        return False
    
class BaseDeclaratorType(DeclaratorType):
    def __init__(self, baseType: TypeSpecifier, qualifiers: TypeQualifier = TypeQualifier()) -> None:
        self.baseType = baseType
        self.qualifiers = qualifiers
        super().__init__()

    def decay(self) -> DeclaratorType:
        # Base types do not decay.
        return BaseDeclaratorType(self.baseType, self.qualifiers)

    def _internal_str(self) -> str:
        return f"{self.qualifiers}{self.baseType}"
    
    def __eq__(self, other):
        if not isinstance(other, BaseDeclaratorType):
            return False
        return self.baseType == other.baseType and self.qualifiers == other.qualifiers
    
    def _internal_copy(self) -> BaseDeclaratorType:
        return BaseDeclaratorType(self.baseType, self.qualifiers)
    
    def unqualify(self) -> BaseDeclaratorType:
        ret = self.copy()
        ret.qualifiers = TypeQualifier()
        return ret

    def getTypeQualifiers(self) -> TypeQualifier:
        return self.qualifiers

    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        self.qualifiers = newQualifiers

class PointerDeclaratorType(DeclaratorType):
    def __init__(self, declarator: DeclaratorType, qualifiers: TypeQualifier = TypeQualifier()) -> None:
        self.declarator = declarator
        self.qualifiers = qualifiers
        super().__init__()

    def decay(self) -> DeclaratorType:
        # Pointers do not decay.
        return PointerDeclaratorType(self.declarator, self.qualifiers)
    
    def _internal_copy(self) -> PointerDeclaratorType:
        return PointerDeclaratorType(self.declarator, self.qualifiers)
        
    def _internal_str(self) -> str:
        return f"{self.qualifiers}pointer to {self.declarator}"
    
    def __eq__(self, other):
        if not isinstance(other, PointerDeclaratorType):
            return False
        return self.declarator == other.declarator and self.qualifiers == other.qualifiers

    def unqualify(self) -> PointerDeclaratorType:
        ret = self.copy()
        ret.qualifiers = TypeQualifier()
        return ret
    
    def getTypeQualifiers(self) -> TypeQualifier:
        return self.qualifiers
    
    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        self.qualifiers = newQualifiers

class ArrayDeclaratorType(DeclaratorType):
    def __init__(self, declarator: DeclaratorType, size: int) -> None:
        self.declarator = declarator
        self.size = size
        super().__init__()

    def decay(self) -> DeclaratorType:
        # An array in C decays to pointer to their first element.
        # For example, int[3][4] decays to int (*)[4].
        # Another example, *int[4] decays to int**.
        return PointerDeclaratorType(self.declarator, self.getTypeQualifiers())
    
    def _internal_copy(self) -> ArrayDeclaratorType:
        return ArrayDeclaratorType(self.declarator, self.size)

    def _internal_str(self) -> str:
        return f"array[{self.size}] of {self.declarator}"
    
    def __eq__(self, other):
        if not isinstance(other, ArrayDeclaratorType):
            return False
        return self.declarator == other.declarator and self.size == other.size

    def unqualify(self) -> ArrayDeclaratorType:
        return ArrayDeclaratorType(self.declarator.unqualify(), self.size)
    
    def getTypeQualifiers(self) -> TypeQualifier:
        return self.declarator.getTypeQualifiers()
    
    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        self.qualifiers = newQualifiers

class FunctionDeclaratorType(DeclaratorType):
    def __init__(self, params: list[ParameterInformation], declarator: DeclaratorType) -> None:
        self.params = params
        self.returnDeclarator = declarator
        super().__init__()

    def decay(self) -> DeclaratorType:
        raise ValueError()

    def _internal_copy(self) -> FunctionDeclaratorType:
        return FunctionDeclaratorType(self.params, self.returnDeclarator)

    def _internal_str(self) -> str:
        paramStrings = [str(p) for p in self.params]
        return f"{self.returnDeclarator}({', '.join(paramStrings)})"
    
    def __eq__(self, other):
        if not isinstance(other, FunctionDeclaratorType):
            return False
        return self.returnDeclarator == other.returnDeclarator and all([p1 == p2 for p1, p2 in zip(self.params, other.params)])

    def unqualify(self) -> FunctionDeclaratorType:
        raise ValueError()
    
    def getTypeQualifiers(self) -> TypeQualifier:
        raise ValueError()

    def setTypeQualifiers(self, newQualifiers: TypeQualifier):
        raise ValueError()
    
@dataclass
class DeclaratorInformation:
    name: str
    type: DeclaratorType
    params: list[ParameterInformation]

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
STATIC EVALUATION
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

class StaticEvalMsgType(enum.Enum):
    # Ordered by priority.
    SUCCESSFUL = enum.auto()
    WARNING = enum.auto()
    ERROR = enum.auto()

@dataclass
class StaticEvalMsg:
    msgType: StaticEvalMsgType
    msg: str = ""

    def __bool__(self) -> bool:
        return self.msgType != StaticEvalMsgType.SUCCESSFUL
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, StaticEvalMsg):
            raise ValueError()
        return self.msgType.value < other.msgType.value
    
    # This is wrongly written, but it does not conflict with the raise keyword.
    def rise(self, warnFunction, errorFunction):
        if self.msgType == StaticEvalMsgType.WARNING:
            warnFunction(self.msg)
        elif self.msgType == StaticEvalMsgType.ERROR:
            errorFunction(self.msg)
    
    @staticmethod
    def WARNING(msg: str) -> StaticEvalMsg:
        return StaticEvalMsg(StaticEvalMsgType.WARNING, msg)

    @staticmethod
    def ERROR(msg: str) -> StaticEvalMsg:
        return StaticEvalMsg(StaticEvalMsgType.ERROR, msg)

EVAL_OK = StaticEvalMsg(StaticEvalMsgType.SUCCESSFUL)

# Used to calculate values during compilation using C-like behavior. 
class StaticEvaluation:
    @staticmethod
    def parseValue(t: TypeSpecifier, val: str|int|float) -> tuple[int|float, StaticEvalMsg]:
        ret: float|int
        retStatus = EVAL_OK
        overflow = False

        if t in (TypeSpecifier.DOUBLE, TypeSpecifier.FLOAT):
            try:
                ret = float(val)
            except Exception as e:
                ret = 0.0
                retStatus = StaticEvalMsg.ERROR(str(e))

            if t == TypeSpecifier.FLOAT:
                ret = struct.unpack('f', struct.pack('f', float(val)))[0]
        else:
            try:
                ret = int(val)
            except:
                try:
                    ret = int(float(val))
                except Exception as e:
                    ret = 0
                    retStatus = StaticEvalMsg.ERROR(str(e))

            if t in (TypeSpecifier.CHAR, TypeSpecifier.SIGNED_CHAR):
                overflow = ret >= 0x80
                ret &= 0xFF
                if ret >= 0x80:
                    ret -= 0x100
            elif t == TypeSpecifier.UCHAR:
                overflow = (ret >= 0x100)
                ret &= 0xFF
            elif t == TypeSpecifier.SHORT:
                overflow = ret >= 0x8000
                ret &= 0xFFFF
                if ret >= 0x8000:
                    ret -= 0x10000
            elif t == TypeSpecifier.USHORT:
                overflow = ret >= 0x10000
                ret &= 0xFFFF
            elif (t == TypeSpecifier.INT) or (t.name == "ENUM"):
                overflow = ret >= 0x80000000
                ret &= 0xFFFFFFFF
                if ret >= 0x80000000:
                    ret -= 0x100000000
            elif t == TypeSpecifier.UINT:
                overflow = ret >= 0x100000000
                ret &= 0xFFFFFFFF
            elif t == TypeSpecifier.LONG:
                overflow = ret >= 0x8000000000000000
                ret &= 0xFFFFFFFFFFFFFFFF
                if ret >= 0x8000000000000000:
                    ret -= 0x10000000000000000
            elif t == TypeSpecifier.ULONG:
                overflow = ret >= 0x10000000000000000
                ret &= 0xFFFFFFFFFFFFFFFF
            else:
                raise ValueError()

        if overflow:
            retStatus = StaticEvalMsg.WARNING(f"{t} {val} overflows to {ret}")
        
        return (ret, retStatus)

    # Static evaluation of operations.
    @staticmethod
    def eval(op: str, tinput: TypeSpecifier, tresult: TypeSpecifier,
             v1: str|int|float, v2: str|int|float|None = None) -> tuple[int|float, StaticEvalMsg]:

        retStatus: StaticEvalMsg = EVAL_OK

        # Evaluate v1 and v2 using tinput.
        v1, w1 = StaticEvaluation.parseValue(tinput, v1)
        if w1 > retStatus:
            retStatus = w1

        if v2 is not None:
            v2, w2 = StaticEvaluation.parseValue(tinput, v2)
            if w2 > retStatus:
                retStatus = w2
        
        # Operate.
        evalStatus: StaticEvalMsg = EVAL_OK
        # Unary operations.
        if op == "~":
            if isinstance(v1, float):
                raise ValueError("Cannot calculate bitwise complement of float")
            ret = ~v1
        elif op == "-" and v2 is None:
            # To differentiate between the binary expression, v2 must be None.
            ret = -v1
        elif op == "!":
            ret = 0 if v1 else 1
        elif op == "++":
            ret = v1 + 1
        elif op == "--":
            ret = v1 - 1
        
        # Binary operations.
        else:
            if v2 is None:
                raise ValueError(f"Null operand received in binary evaluation of {op}")
            
            elif op == "*":
                ret = v1 * v2
            elif op == "/":
                if v2 == 0:
                    evalStatus = StaticEvalMsg.ERROR("Division by zero")

                if isinstance(v2, float):
                    v1Zero    = v1 == 0.0
                    v1Neg     = math.copysign(1, v1) == -1.0
                    v2Zero    = v2 == 0.0
                    v2Neg     = math.copysign(1, v2) == -1.0
                    
                    if v1Zero and v2Zero:
                        ret = math.nan
                    elif v1Zero and not v2Zero:
                        if v1Neg:
                            ret = -0.0 if v2 > 0 else 0.0
                        else:
                            ret = 0.0 if v2 > 0 else -0.0
                    elif not v1Zero and v2Zero:
                        if v2Neg:
                            ret = -math.inf if v1 > 0 else math.inf
                        else:
                            ret = math.inf if v1 > 0 else -math.inf
                    else:
                        ret = v1 / v2
                else:
                    if v2 == 0:
                        ret = 0xFFFFFFFFFFFFFFFF
                    else:
                        # Python integer division // rounds towards negative infinity, so if we get
                        # -1.8, this is rounded to -2, and not to -1 like in C.
                        ret = v1 // v2
                        if ret < 0 and v1 % v2 != 0:
                            ret += 1
            elif op == "%":
                if v2 == 0:
                    ret = 0
                    evalStatus = StaticEvalMsg.ERROR("Division by zero")
                else:
                    # In C, % behave differently to Python.
                    ret = v1 % v2
                    # If the signs are different and the remainder is non-zero,
                    # adjust to match the sign of the dividend (a)
                    if (v1 < 0) != (v2 < 0) and ret != 0:
                        ret -= v2
            elif op == "+":
                ret = v1 + v2
            elif op == "-":
                ret = v1 - v2
            elif op == "<<":
                if isinstance(v1, float) or isinstance(v2, float):
                    raise ValueError()
                ret = v1 << v2
            elif op == ">>":
                if isinstance(v1, float) or isinstance(v2, float):
                    raise ValueError()
                ret = v1 >> v2
            elif op == ">":
                ret = 1 if v1 > v2 else 0
            elif op == ">=":
                ret = 1 if v1 >= v2 else 0
            elif op == "<":
                ret = 1 if v1 < v2 else 0
            elif op == "<=":
                ret = 1 if v1 <= v2 else 0
            elif op == "==":
                ret = 1 if v1 == v2 else 0
            elif op == "!=":
                ret = 1 if v1 != v2 else 0
            elif op == "&":
                if isinstance(v1, float) or isinstance(v2, float):
                    raise ValueError()
                ret = v1 & v2
            elif op == "^":
                if isinstance(v1, float) or isinstance(v2, float):
                    raise ValueError()
                ret = v1 ^ v2
            elif op == "|":
                if isinstance(v1, float) or isinstance(v2, float):
                    raise ValueError()
                ret = v1 | v2
            elif op == "&&":
                ret = 1 if v1 and v2 else 0
            elif op == "||":
                ret = 1 if v1 or v2 else 0
            else:
                raise ValueError(f"Invalid operation {op} found during static evaluation")

            if evalStatus > retStatus:
                retStatus = evalStatus

        # Evaluate the result using tresult.
        ret, w_end = StaticEvaluation.parseValue(tresult, ret)
        if w_end > retStatus:
            retStatus = w_end

        return (ret, retStatus)

    # Similar to above but using declarators instead of type specifiers.
    @staticmethod
    def evalDecl(op: str, tinput: DeclaratorType, tresult: DeclaratorType,
                v1: str|int|float, v2: str|int|float|None = None) -> tuple[int|float, StaticEvalMsg]:
        

        if isinstance(tinput, BaseDeclaratorType):
            inputBaseType = tinput.baseType
        elif isinstance(tinput, PointerDeclaratorType) and int(v1) == 0:
            # This is a null pointer, which can be constant folded.
            # TODO: this is only for 64 bit systems.
            inputBaseType = TypeSpecifier.ULONG
        else:
            raise ValueError(f"Invalid input declarator {tinput} in static evaluation")
        
        if not isinstance(tresult, BaseDeclaratorType):
            raise ValueError(f"Invalid result declarator {tresult} in static evaluation")

        return StaticEvaluation.eval(op, inputBaseType, tresult.baseType, v1, v2)